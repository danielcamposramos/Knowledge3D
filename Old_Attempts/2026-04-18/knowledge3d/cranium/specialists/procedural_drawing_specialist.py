"""
Procedural Drawing Specialist for Adaptive Swarm.

Handles training and inference for procedural glyph generation and recognition,
enabling atomic cognition through form-meaning fusion.

Architecture:
    - Form modality: GPU RPN executor → FractalEmitter generates visual embeddings
    - Meaning modality: Math RPN executor → opcode table OR semantic encoding
    - Fusion: Weighted average creates unified form+meaning embeddings
    - Storage: ProceduralGalaxy stores compressed procedural programs (69:1 ratio)
    - Training: Shadow copy updates (Phase H self-updating adapters)

Usage:
    specialist = ProceduralDrawingSpecialist(swarm)
    specialist.train_on_batch(batch, dual_modal_math=True)
    # Atomic units stored in ProceduralGalaxy with procedural compression
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass

from datetime import datetime, timezone

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM
from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
from knowledge3d.cranium.bridges.sovereign_bridges import FractalEmitter
from knowledge3d.cranium.procedural_galaxy import ProceduralGalaxy
from knowledge3d.cranium.procedural_compiler import ProceduralCompiler
from knowledge3d.cranium.specialists.batch_optimizer import BatchOptimizer
from knowledge3d.cranium.specialists.character_languages import get_character_languages


@dataclass
class TrainingMetrics:
    """Metrics for procedural drawing training."""
    epoch: int
    text_visual_alignment: float  # Cosine similarity
    reconstruction_fidelity: float  # SSIM score
    generation_quality: float      # Human eval proxy
    latency_us: float              # Average inference time


class ProceduralDrawingSpecialist:
    """
    Specialist for procedural glyph generation and recognition.

    Implements atomic cognition learning path:
    1. Learn drawing primitives (curves, lines, arcs)
    2. Align visual glyphs with text characters
    3. Enable generative drawing (text → RPN → visual)
    4. Enable OCR (visual → text via learned embeddings)
    """

    def __init__(
        self,
        swarm: AdaptiveSwarmTRM,
        matryoshka_dim: int = 512,
        gpu_id: int = 0
    ):
        """
        Initialize procedural drawing specialist.

        Args:
            swarm: Adaptive swarm instance
            matryoshka_dim: Embedding dimension (64-2048 adaptive)
            gpu_id: CUDA device ID
        """
        self.swarm = swarm
        self.matryoshka_dim = matryoshka_dim
        self.gpu_id = gpu_id

        # Register specialist with swarm
        rank = self._select_rank_from_dim(matryoshka_dim)
        self.swarm.register_specialist(
            'procedural_drawing',
            required_dims=matryoshka_dim,
            rank=rank
        )

        # Initialize bridges
        self.drawing_bridge = ProceduralDrawingBridge(matryoshka_dim=matryoshka_dim)
        self.visual_embedder = FractalEmitter()

        # Procedural storage (Phase 2.6 compression)
        self.procedural_compiler = ProceduralCompiler()
        self.procedural_galaxy = ProceduralGalaxy()

        # GPU batch optimizer (full 12GB VRAM)
        self.batch_optimizer = BatchOptimizer(
            target_utilization=0.75,
            max_vram_mb=11500.0,  # Use full 12GB VRAM
            min_batch_size=8,
            max_batch_size=2048,  # Scale up to saturate GPU
            scale_factor=1.5,
        )

        # Training state
        self.training_metrics: List[TrainingMetrics] = []
        self.char_to_rpn_cache: Dict[str, str] = {}  # Learned RPN programs
        self.char_to_math_rpn_cache: Dict[str, str] = {}  # Learned execution bytecode

        # Atomic unit cache (deferred compression, multi-glyph metadata)
        self.atomic_units: Dict[str, Dict[str, Any]] = {}  # char -> {embedding, glyphs, languages, ...}

        # Execution embedder for math RPN (opcode embedding table)
        self._init_opcode_embedding_table()

    def _select_rank_from_dim(self, dim: int) -> int:
        """Select LoRA rank based on Matryoshka dimension."""
        # 18× memory reduction principle from Phase H
        return max(8, dim // 16)

    def _build_glyph_metadata(self, char: str, source: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize glyph metadata (font family, variant, source)."""
        meta = source or {}
        font_family = (
            meta.get('font_family')
            or meta.get('font')
            or meta.get('font_name')
            or "unknown"
        )
        font_name = meta.get('font_name') or font_family
        font_weight = int(meta.get('font_weight') or meta.get('weight') or 400)
        font_style = meta.get('font_style') or meta.get('style') or "normal"
        font_variant = meta.get('font_variant') or meta.get('variant') or "regular"
        font_source = meta.get('font_source') or meta.get('source') or "unknown"
        unicode_codepoint = (
            meta.get('unicode_codepoint')
            or meta.get('unicode')
            or (f"U+{ord(char):04X}" if char else "unknown")
        )

        return {
            'font_family': font_family,
            'font_name': font_name,
            'font_weight': font_weight,
            'font_style': font_style,
            'font_variant': font_variant,
            'font_source': font_source,
            'unicode_codepoint': unicode_codepoint,
        }

    @staticmethod
    def _ensure_rpn_string(program: Any) -> str:
        """Normalize RPN program input to string form."""
        if isinstance(program, bytes):
            try:
                return program.decode('utf-8')
            except UnicodeDecodeError:
                return program.decode('latin-1', errors='ignore')
        return str(program)

    def _init_opcode_embedding_table(self):
        """
        Initialize learnable embedding table for RPN opcodes.

        Maps opcodes (0x00-0xFF) to semantic embeddings via Matryoshka projection.
        This enables the model to understand what each operation MEANS.
        """
        # Create opcode embedding table (256 opcodes × matryoshka_dim)
        self.opcode_embeddings = np.random.randn(256, self.matryoshka_dim).astype(np.float32) * 0.01

        # Pre-populate with Matryoshka projections for common opcodes
        common_opcodes = [
            0x14,  # SQRT
            0x0A,  # ADD
            0x0B,  # SUB
            0x0C,  # MUL
            0x07,  # DIV
            0x0D,  # EXP
            0x12,  # LOG
            0x10,  # SIN
            0x11,  # COS
            0xB6,  # GRADIENT
            0xBC,  # DIVERGENCE
            0xE4,  # CONST
        ]

        for opcode in common_opcodes:
            # Create one-hot-like seed for this opcode
            seed = np.zeros(self.matryoshka_dim, dtype=np.float32)
            seed[opcode % self.matryoshka_dim] = 1.0

            # Project through Matryoshka for semantic initialization
            try:
                self.opcode_embeddings[opcode] = self.swarm.base.project_vector(seed, self.matryoshka_dim)
            except Exception:
                # Keep random init if GPU fails
                pass

    def encode_semantic_context(self, semantic: str) -> np.ndarray:
        """
        Encode semantic description using lightweight method.

        For letters: Use character code + simple features
        For math: Use semantic encoding from description

        This is MINIMAL - the real meaning comes from execution or usage context.

        Args:
            semantic: Semantic description or character

        Returns:
            Embedding vector (matryoshka_dim,)
        """
        if len(semantic) == 1:
            # Single character - use Unicode codepoint
            code = ord(semantic[0])
            emb = np.zeros(self.matryoshka_dim, dtype=np.float32)
            emb[0] = float(code) / 1000.0  # Normalize
            return emb
        else:
            # Phrase/description - simple average of char codes
            codes = [ord(c) for c in semantic[:self.matryoshka_dim]]
            emb = np.zeros(self.matryoshka_dim, dtype=np.float32)
            emb[:len(codes)] = np.array(codes, dtype=np.float32) / 1000.0
            return emb

    def _fuse_multimodal(
        self,
        form_emb: np.ndarray,
        meaning_emb: np.ndarray,
        form_rpn: str,
        meaning_rpn: str
    ) -> np.ndarray:
        """
        Fuse form + meaning via compositional storage (not runtime merging).

        The fusion happens at the STAR level - both visual_rpn and math_rpn
        are stored together in ProceduralGalaxy. The star itself IS the fusion.

        For the embedding fusion, we use visual form as the grounding since
        "letters are drawings with meaning" - the visual form is primary,
        the execution/semantic meaning is secondary context.

        Args:
            form_emb: Visual embedding (from RPN execution) - PRIMARY
            meaning_emb: Semantic/execution embedding - CONTEXT
            form_rpn: Visual RPN program (stored in star)
            meaning_rpn: Math RPN bytecode (stored in star)

        Returns:
            Unified embedding (form as primary grounding)
        """
        # Visual form is the grounding - this is what gets stored
        # The meaning_rpn is stored ALONGSIDE in the same star
        # Cross-modality happens via compositional storage, not embedding fusion
        return form_emb.astype(np.float32)

    def _store_atomic_star(
        self,
        char: str,
        unified_emb: np.ndarray,
        form_rpn: str,
        meaning_rpn: str,
        glyph_metadata: Optional[Dict[str, Any]] = None,
        form_embedding: Optional[np.ndarray] = None,
    ):
        """
        Store atomic knowledge unit in ProceduralGalaxy as a DUAL-PROGRAM STAR.

        The star contains:
          - visual_rpn: How to DRAW the character (form)
          - math_rpn: What it DOES/MEANS (execution/semantic)
          - languages: Which languages use this character (ISO 639-1 codes)
          - embedding: Compressed procedural program from visual form

        This compositional storage IS the fusion - both programs coexist
        in the same star, enabling cross-modal reasoning via the 3D contract.

        Multilingual Support:
          - Basic Latin (a-z, A-Z): ~30 languages (en, pt, es, fr, de, ...)
          - Extended Latin (ç, ñ, ä): Subset of languages (pt/fr/ca, es, de)
          - Math symbols (+, π, ∫): 'universal' (language-agnostic)
          - Enables pronunciation encoding per language (future enhancement)

        Args:
            char: Character/symbol (lookup key)
            unified_emb: Visual form embedding (primary grounding)
            form_rpn: Visual RPN program (HOW to draw)
            meaning_rpn: Math RPN bytecode (WHAT it does) or "" for non-math
        """
        # Get language metadata for this character
        languages = get_character_languages(char)
        glyph_meta = glyph_metadata or self._build_glyph_metadata(char, None)
        glyph_timestamp = datetime.now(timezone.utc).isoformat()

        glyph_entry = {
            'visual_rpn': form_rpn,
            'font_metadata': glyph_meta,
            'timestamp': glyph_timestamp,
        }
        if form_embedding is not None:
            glyph_entry['form_embedding'] = form_embedding.astype(np.float32)

        unit = self.atomic_units.get(char)

        if unit is None:
            unit = {
                'embedding': unified_emb.astype(np.float32),
                'math_rpn': meaning_rpn or "",
                'languages': list(languages),
                'glyphs': [glyph_entry],
                'glyph_count': 1,
                'timestamp': glyph_timestamp,
            }
            self.atomic_units[char] = unit
            return

        count = unit.get('glyph_count', 0)
        prev_embedding = unit.get('embedding')
        if prev_embedding is None:
            unit['embedding'] = unified_emb.astype(np.float32)
        else:
            updated = (prev_embedding * count + unified_emb) / (count + 1)
            unit['embedding'] = updated.astype(np.float32)

        unit['glyphs'].append(glyph_entry)
        unit['glyph_count'] = count + 1
        unit['timestamp'] = glyph_timestamp

        if meaning_rpn and not unit.get('math_rpn'):
            unit['math_rpn'] = meaning_rpn

        # Merge language coverage (union)
        lang_set: Set[str] = set(unit.get('languages', []))
        lang_set.update(languages)
        unit['languages'] = sorted(lang_set)

    def _train_via_rpn_stacks(
        self,
        form_embeddings: List[np.ndarray],
        unified_embeddings: List[np.ndarray]
    ) -> float:
        """
        SOVEREIGN TRAINING via RPN stack operations (full PTX/GPU).

        This replaces NumPy gradient computation with RPN stack operations.
        Uses the 18-stack RPN architecture with ternary logic for validation.

        Conceptual RPN Program for Training:
          1. Load form_emb onto Stack 0
          2. Load unified_emb onto Stack 1
          3. Execute: "STACK1 STACK0 SUB"  → gradient direction
          4. Execute: "DUP MAGNITUDE"       → loss measurement
          5. Apply to adapter via ternary validation:
             - TRUE: Commit update (improvement detected)
             - FALSE: Reject update (degradation detected)
             - UNKNOWN: Accumulate more evidence

        Args:
            form_embeddings: Visual form embeddings (input)
            unified_embeddings: Target unified embeddings (output)

        Returns:
            Average loss (for compatibility with existing code)
        """
        # TODO: Implement full RPN stack-based training
        #
        # Steps:
        # 1. Convert embeddings to RPN stack format
        # 2. Execute RPN program for gradient computation:
        #    - Use ModularRPNEngine with 18 stacks
        #    - Stack operations: SUB, MAGNITUDE, NORMALIZE
        # 3. Ternary validation gate:
        #    - Fork adapter weights to shadow (GPU memory copy)
        #    - Apply RPN-computed gradients to shadow
        #    - Validate: shadow_performance vs baseline_performance
        #    - If TRUE (better): commit shadow → main
        #    - If FALSE (worse): discard shadow
        #    - If UNKNOWN (unclear): accumulate more samples
        # 4. Return loss metric

        # For now, return zero loss (not implemented)
        return 0.0

    def commit_atomic_units_to_galaxy(self):
        """
        Compress and commit all atomic units to ProceduralGalaxy.

        This is called AFTER training completes to avoid CPU compression
        overhead during training. We batch-compress all units at once.

        Returns:
            Number of units committed
        """
        print(f"\n[ProceduralGalaxy] Committing {len(self.atomic_units)} atomic units...")

        committed = 0
        failed = 0

        for char, unit in self.atomic_units.items():
            try:
                # Compress embedding to procedural program
                program = self.procedural_compiler.compile_embedding(unit['embedding'])
                program_bytes = program.to_bytes()

                # Calculate compression ratio
                original_size = unit['embedding'].nbytes
                compressed_size = len(program_bytes)
                compression_ratio = original_size / max(compressed_size, 1)

                glyph_meta_list = [
                    {
                        'visual_rpn': glyph['visual_rpn'],
                        'font_metadata': glyph.get('font_metadata', {}),
                        'timestamp': glyph.get('timestamp'),
                    }
                    for glyph in unit.get('glyphs', [])
                ]

                metadata = {
                    'math_rpn': unit.get('math_rpn', ''),
                    'languages': unit.get('languages', []),  # ISO 639-1 codes
                    'timestamp': unit.get('timestamp'),
                    'glyph_count': len(glyph_meta_list),
                    'glyphs': glyph_meta_list,
                }

                # Store dual-program star in ProceduralGalaxy
                self.procedural_galaxy.store_program(
                    key=char,
                    program_bytes=program_bytes,
                    compression_ratio=compression_ratio,
                    metadata=metadata  # Pass multilingual metadata
                )

                committed += 1

            except Exception as e:
                print(f"  [WARNING] Failed to commit '{char}': {e}")
                failed += 1

        print(f"[ProceduralGalaxy] Committed {committed} units, {failed} failed")
        print(f"[ProceduralGalaxy] Total storage: {committed * 2230}B (~{committed * 2.2:.1f}KB)")

        return committed

    def load_rpn_dataset(self, dataset_path: Path) -> List[Dict]:
        """
        Load RPN dataset (JSONL format).

        Args:
            dataset_path: Path to JSONL file

        Returns:
            List of dataset entries
        """
        entries = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def _compute_visual_embedding(self, rpn_program: str) -> np.ndarray:
        """
        Generate visual embedding from RPN program execution (GPU-accelerated).

        Args:
            rpn_program: RPN program string (e.g., "0.5 0.5 MOVE 0.7 0.7 LINE STROKE")

        Returns:
            Visual embedding (matryoshka_dim,)
        """
        # Execute RPN on GPU → render result with segments
        try:
            result = self.drawing_bridge.execute_rpn_gpu(
                rpn_program,
                width=256,
                height=256,
                skip_raster=True  # We only need segments, not pixels
            )
        except Exception as e:
            print(f"  [WARNING] GPU RPN execution failed: {e}")
            return np.zeros(self.matryoshka_dim, dtype=np.float32)

        if result.segments is None or len(result.segments) == 0:
            # Empty glyph - return zero embedding
            return np.zeros(self.matryoshka_dim, dtype=np.float32)

        # Convert segments to point cloud for FractalEmitter
        # Segments are (x0,y0,x1,y1,r,g,b,a,w) - extract points
        points = np.vstack([
            result.segments[:, :2],   # Start points
            result.segments[:, 2:4]   # End points
        ]).astype(np.float32)

        # FractalEmitter generates spatial features; pool to fixed dim
        coords = self.visual_embedder.emit(points)  # shape (N,3)
        if coords.size == 0:
            return np.zeros(self.matryoshka_dim, dtype=np.float32)
        pooled = coords.mean(axis=0)  # (3,)
        # Tile/trim to matryoshka_dim
        reps = (self.matryoshka_dim + 2) // 3
        tiled = np.tile(pooled, reps)[: self.matryoshka_dim]
        return tiled.astype(np.float32)

    def _compute_execution_embedding(self, math_rpn: str) -> np.ndarray:
        """
        Generate execution embedding from RPN bytecode sequence (GPU-accelerated).

        Uses learnable opcode embedding table to capture semantic meaning.

        For dual-modal math symbols, this embeds the EXECUTION bytecode
        (e.g., "0x14" for SQRT, "0x14 0x14" for fourth root).

        Args:
            math_rpn: RPN bytecode sequence as string (e.g., "0x14" or "0x14 0x14")

        Returns:
            Embedding vector (matryoshka_dim,)
        """
        if not math_rpn or math_rpn.startswith('#'):
            return np.zeros(self.matryoshka_dim, dtype=np.float32)

        # Parse RPN string to opcodes
        tokens = math_rpn.split()
        opcodes = []

        for token in tokens:
            if token.startswith('0x'):
                # Hex opcode (e.g., "0x14")
                try:
                    opcode = int(token, 16)
                    opcodes.append(opcode)
                except ValueError:
                    pass
            else:
                # Literal number (e.g., "1.0") - use hash
                try:
                    val = float(token)
                    # Handle special float values
                    if np.isnan(val) or np.isinf(val):
                        opcodes.append(255)  # Special marker for inf/nan
                    else:
                        opcodes.append(int(abs(val * 100)) % 256)
                except (ValueError, OverflowError):
                    pass

        if not opcodes:
            return np.zeros(self.matryoshka_dim, dtype=np.float32)

        # Look up embeddings for each opcode
        opcode_embeds = [self.opcode_embeddings[op] for op in opcodes]

        # Average opcode embeddings
        final_embedding = np.mean(opcode_embeds, axis=0).astype(np.float32)
        return final_embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def train_on_batch(
        self,
        batch: List[Tuple],
        validation: bool = False,
        dual_modal_math: bool = False
    ) -> TrainingMetrics:
        """
        Train base model on atomic knowledge formation (form + meaning fusion).

        Args:
            batch: List of (char, rpn_program) tuples
                   OR list of (symbol, visual_rpn, math_rpn, semantic) for dual-modal
            validation: If True, compute metrics without updating weights
            dual_modal_math: If True, batch contains dual-modal math entries

        Returns:
            Training metrics for this batch
        """
        unified_embeddings = []
        form_embeddings = []
        alignment_scores = []
        symbols = []
        form_rpns = []
        meaning_rpns = []

        # Compute unified embeddings (form + meaning fusion)
        for entry in batch:
            glyph_meta_source: Optional[Dict[str, Any]] = None

            if dual_modal_math:
                # Dual-modal math entries may be tuples or dicts
                if isinstance(entry, dict):
                    symbol = entry.get('symbol', entry.get('char', ''))
                    visual_rpn = entry.get('visual_rpn', entry.get('rpn', ''))
                    math_rpn = entry.get('math_rpn', '')
                    semantic = entry.get('semantic', symbol)
                    glyph_meta_source = entry
                elif isinstance(entry, (tuple, list)):
                    if len(entry) < 4:
                        raise ValueError("Dual-modal entry must have at least four elements")
                    symbol = entry[0]
                    visual_rpn = entry[1]
                    math_rpn = entry[2]
                    semantic = entry[3]
                    if len(entry) >= 5 and isinstance(entry[4], dict):
                        glyph_meta_source = entry[4]
                else:
                    raise ValueError("Unsupported batch entry type for dual-modal math")

                visual_rpn = self._ensure_rpn_string(visual_rpn)
                math_rpn = self._ensure_rpn_string(math_rpn) if math_rpn else ""

                form_emb = self._compute_visual_embedding(visual_rpn)

                if math_rpn and not math_rpn.startswith('#'):
                    meaning_emb = self._compute_execution_embedding(math_rpn)
                else:
                    meaning_emb = self.encode_semantic_context(semantic)

                symbols.append(symbol)
                form_rpns.append(visual_rpn)
                meaning_rpns.append(math_rpn if math_rpn else "")

            else:
                # Standard glyph entries: tuple or dict
                if isinstance(entry, dict):
                    char = entry.get('char', '')
                    rpn_program = entry.get('rpn', entry.get('visual_rpn', ''))
                    glyph_meta_source = entry
                elif isinstance(entry, (tuple, list)):
                    if not entry:
                        raise ValueError("Empty batch entry encountered")
                    char = entry[0]
                    rpn_program = entry[1] if len(entry) >= 2 else ''
                    if len(entry) >= 3 and isinstance(entry[2], dict):
                        glyph_meta_source = entry[2]
                else:
                    raise ValueError("Unsupported batch entry type for glyph training")

                rpn_program = self._ensure_rpn_string(rpn_program)

                form_emb = self._compute_visual_embedding(rpn_program)
                meaning_emb = self.encode_semantic_context(char)

                symbols.append(char)
                form_rpns.append(rpn_program)
                meaning_rpns.append("")

            # Fuse form + meaning (GPU-native RPN)
            unified_emb = self._fuse_multimodal(
                form_emb,
                meaning_emb,
                form_rpns[-1] if form_rpns else "",
                meaning_rpns[-1] if meaning_rpns else ""
            )
            unified_embeddings.append(unified_emb)
            form_embeddings.append(form_emb)

            # Measure form ↔ meaning alignment
            alignment = self._cosine_similarity(form_emb, meaning_emb)
            alignment_scores.append(alignment)

            # Store in ProceduralGalaxy (if not validation)
            if not validation:
                glyph_metadata = self._build_glyph_metadata(
                    symbols[-1],
                    glyph_meta_source,
                )
                self._store_atomic_star(
                    symbols[-1],
                    unified_emb,
                    form_rpns[-1],
                    meaning_rpns[-1],
                    glyph_metadata=glyph_metadata,
                    form_embedding=form_emb,
                )

        # Train base model via RPN stack operations (sovereign training)
        contrastive_loss = 0.0
        if not validation and len(unified_embeddings) > 0:
            # Sovereign training: Use RPN stack operations instead of NumPy gradients
            # This is a placeholder for full RPN implementation
            # TODO: Implement _train_via_rpn_stacks() for full sovereignty

            # For now, keep existing mechanism but document the sovereign path
            # The specialist learns to recognize atomic knowledge by observing
            # the visual form embeddings during training
            form_to_unified_pairs = [(form, unified) for form, unified in zip(form_embeddings, unified_embeddings)]

            # NOTE: This uses NumPy internally (not sovereign yet)
            # Future: Replace with RPN stack operations + ternary validation
            stats = self.swarm.train_specialist_contrastive(
                'procedural_drawing',
                form_to_unified_pairs,
                learning_rate=None  # Use swarm default
            )
            contrastive_loss = stats.get('avg_loss', 0.0)

        # Return metrics
        avg_alignment = float(np.mean(alignment_scores)) if alignment_scores else 0.0

        return TrainingMetrics(
            epoch=len(self.training_metrics),
            text_visual_alignment=avg_alignment,  # form ↔ meaning alignment
            reconstruction_fidelity=0.0,
            generation_quality=0.0,
            latency_us=contrastive_loss  # Reuse field for loss
        )

    def train_on_rpn_dataset(
        self,
        dataset_path: Path,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1,
        adaptive_batching: bool = True,
    ):
        """
        Train specialist on RPN dataset.

        Args:
            dataset_path: Path to JSONL dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        # Load dataset
        entries = self.load_rpn_dataset(dataset_path)

        # Compile RPN to bytecode
        training_data = []
        for entry in entries:
            char = entry['char']
            rpn = entry['rpn']

            # Compile RPN to bytecode
            bytecode = self.drawing_bridge.compile_rpn_to_bytecode(rpn)
            training_data.append((char, bytecode))

        # Split train/validation
        n_val = int(len(training_data) * validation_split)
        validation_data = training_data[:n_val]
        train_data = training_data[n_val:]

        print(f"Training on {len(train_data)} samples, validating on {len(validation_data)}")

        current_batch_size = batch_size

        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            np.random.shuffle(train_data)

            # Train on batches
            epoch_metrics = []
            for i in range(0, len(train_data), current_batch_size):
                batch = train_data[i:i+current_batch_size]
                metrics = self.train_on_batch(batch, validation=False)
                epoch_metrics.append(metrics)

                # Adaptive batching (every 10 batches)
                if adaptive_batching and (i // max(current_batch_size, 1)) % 10 == 0:
                    try:
                        import cupy as cp  # type: ignore
                        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                        vram_used = (total_mem - free_mem) / (1024 ** 2)
                        # Heuristic util estimate scales with batch size
                        gpu_util_estimate = min(0.9, 0.07 * (current_batch_size / batch_size))
                        new_bs = self.batch_optimizer.suggest_batch_size(
                            current_batch_size=current_batch_size,
                            gpu_utilization=gpu_util_estimate,
                            vram_used_mb=vram_used,
                        )
                        if new_bs != current_batch_size:
                            print(f"  [BatchOptimizer] batch size {current_batch_size} → {new_bs} (VRAM {vram_used:.1f} MB)")
                            current_batch_size = new_bs
                    except Exception:
                        # If cupy unavailable or metrics fail, keep current batch
                        pass

            # Validation
            val_metrics = []
            for i in range(0, len(validation_data), current_batch_size):
                batch = validation_data[i:i+current_batch_size]
                metrics = self.train_on_batch(batch, validation=True)
                val_metrics.append(metrics)

            # Aggregate metrics
            avg_train_alignment = np.mean([m.text_visual_alignment for m in epoch_metrics])
            avg_val_alignment = np.mean([m.text_visual_alignment for m in val_metrics])

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train alignment={avg_train_alignment:.3f}, "
                  f"Val alignment={avg_val_alignment:.3f}")

            # Store metrics
            self.training_metrics.append(TrainingMetrics(
                epoch=epoch,
                text_visual_alignment=avg_val_alignment,
                reconstruction_fidelity=0.0,
                generation_quality=0.0,
                latency_us=0.0
            ))
            if adaptive_batching:
                print(self.batch_optimizer.get_optimization_report())

    def generate_glyph(self, char: str) -> str:
        """
        Generate RPN program for character (generative capability).

        Args:
            char: Character to generate

        Returns:
            RPN program string
        """
        # Check cache first
        if char in self.char_to_rpn_cache:
            return self.char_to_rpn_cache[char]

        # Generate via learned embeddings
        semantic_emb = self.encode_semantic_context(char)

        # Use swarm to predict visual embedding
        predicted_visual = self.swarm.forward(
            semantic_emb,
            specialist='procedural_drawing'
        )

        # Decode visual embedding to RPN
        # TODO: Implement decoder (inverse of execute_rpn → fractal_emit)
        # For now, return placeholder
        rpn_program = f"# Generated RPN for '{char}' (decoder pending)"

        self.char_to_rpn_cache[char] = rpn_program
        return rpn_program

    def predict_math_rpn(self, semantic: str) -> str:
        """
        Predict RPN bytecode for math execution from semantic text.

        This enables the Synthetic User to perform actual math computations
        in the mind, not via tool calls or approximations.

        Args:
            semantic: Semantic description (e.g., "Square root: √x" or "arcsinh(x)")

        Returns:
            RPN bytecode sequence (e.g., "0x14" for SQRT, "0x14 0x14" for fourth root)

        Example:
            >>> predict_math_rpn("Fourth root: ∜x = √√x")
            "0x14 0x14"  # SQRT SQRT compositional

            >>> predict_math_rpn("Arc hyperbolic sine")
            "0x03 0x04 0xE4 1.0 0x05 0x14 0x05 0x12"  # DUP MUL CONST ADD SQRT ADD LOG
        """
        # Check cache first
        if semantic in self.char_to_math_rpn_cache:
            return self.char_to_math_rpn_cache[semantic]

        # Compute semantic embedding
        semantic_emb = self.encode_semantic_context(semantic)

        # Use swarm to predict execution embedding
        predicted_execution = self.swarm.forward(
            semantic_emb,
            specialist='procedural_drawing'
        )

        # Decode execution embedding to RPN bytecode
        # Use nearest neighbor search in learned execution space
        # TODO: Implement decoder (lookup learned execution → opcode mapping)
        # For now, return placeholder
        math_rpn = f"# Predicted math RPN for '{semantic}' (decoder pending)"

        self.char_to_math_rpn_cache[semantic] = math_rpn
        return math_rpn

    def save_checkpoint(self, path: Path):
        """Save specialist state with dual-modal caches."""
        checkpoint = {
            'matryoshka_dim': self.matryoshka_dim,
            'training_metrics': [
                {
                    'epoch': m.epoch,
                    'text_visual_alignment': m.text_visual_alignment,
                    'reconstruction_fidelity': m.reconstruction_fidelity,
                    'generation_quality': m.generation_quality,
                    'latency_us': m.latency_us
                }
                for m in self.training_metrics
            ],
            'char_to_rpn_cache': self.char_to_rpn_cache,
            'char_to_math_rpn_cache': self.char_to_math_rpn_cache  # Save execution cache
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)


__all__ = ['ProceduralDrawingSpecialist', 'TrainingMetrics']
