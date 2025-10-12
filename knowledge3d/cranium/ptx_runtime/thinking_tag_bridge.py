import ctypes
import numpy as np
import logging
import os

# Kimi's zero-copy strategy: local imports maintain GPU pointers without host-device copies
from .sovereign.loader import gpu_malloc, memcpy_htod, memcpy_dtoh, launch_kernel
from .modular_rpn_engine import ModularRPNEngine, RPNProgram
from .galaxy_resonance_engine import ResonanceField
from .temporal_reasoning import TemporalReasoning
from .graph_crystallizer import GraphCrystallizer
from .vector_resonator import VectorResonator
from .galaxy_memory_updater import GalaxyMemoryUpdater
from .atomic_fission_fusion import AtomicFissionFusion
from .latency_guard import LatencyGuard
from .adaptive_sparsity_engine import AdaptiveSparsityEngine
from .cross_modal_resonance_engine import CrossModalResonanceEngine
from .fractal_emitter import FractalEmitter
from .galaxy_visualizer import GalaxyVisualizer

logger = logging.getLogger(__name__)
tag_names = [f"tag_{i}" for i in range(100)]

class ThinkingTagOutput:
    def __init__(self, probs, confidence_rays, uncertainty, coherence_scores):
        self.probs = probs
        self.confidence_rays = confidence_rays
        self.uncertainty = uncertainty
        self.coherence_scores = coherence_scores
        self.tags = []

class ThinkingTagBridge:
    MODE_FULL_TEMPORAL = 0
    MODE_SPATIAL_ONLY = 1
    MODE_DEBUG_VALIDATION = 2

    def __init__(self):
        self.rpn_engine = ModularRPNEngine()
        self.resonance_field = ResonanceField()
        self.temporal_reasoning = TemporalReasoning()
        self.graph_crystallizer = GraphCrystallizer()
        self.vector_resonator = VectorResonator()
        self.galaxy_memory_updater = GalaxyMemoryUpdater()
        self.atomic_fission_fusion = AtomicFissionFusion()
        self.latency_guard = LatencyGuard()
        self.fractal_emitter = FractalEmitter()
        
        # GLM's enhancements
        self.adaptive_sparsity = AdaptiveSparsityEngine(
            self.vector_resonator, 
            self.atomic_fission_fusion
        )
        self.cross_modal_engine = CrossModalResonanceEngine(self.fractal_emitter)

        # Visualization (optional)
        self.visualizer = None
        if os.getenv("K3D_ENABLE_THINKING_TAG_VISUALIZATION", "0").lower() in ("1", "true"):
            self.visualizer = GalaxyVisualizer(self.resonance_field)

        # Sovereign buffers
        self.ema_buffer = gpu_malloc(256)
        self.mode_buffer = gpu_malloc(4)
        self.temp_buffers = gpu_malloc(2048)
        self.cache_buffer = gpu_malloc(1024 * 1024)  # 1MB cache

        # Initialize
        self._reset_ema_buffer_gpu()
        self.set_mode(0)
        self._warm_resonance_cache()
        
        # Cache for fallback
        self._cached_spatial_prog = None

    def _reset_ema_buffer_gpu(self):
        """Sovereign GPU-side memset for EMA buffer"""
        zero_kernel_ptx = """
        .version 7.8
        .target sm_86
        .address_size 64
        .visible .entry zero_fill(.param .u64 buf, .param .u32 bytes) {
            .reg .u64 %ptr, %end;
            .reg .u32 %cnt;
            ld.param.u64 %ptr, [buf];
            ld.param.u32 %cnt, [bytes];
            shl.b32 %cnt, %cnt, 2;
            add.u64 %end, %ptr, %cnt;
        $L_loop:
            setp.ge.u64 %p1, %ptr, %end;
            @%p1 bra $L_done;
            st.global.u32 [%ptr], 0;
            add.u64 %ptr, %ptr, 4;
            bra $L_loop;
        $L_done:
            ret;
        }
        """
        # Save and launch zero kernel
        ptx_path = "knowledge3d/cranium/ptx/zero_fill.ptx"
        with open(ptx_path, "w") as f:
            f.write(zero_kernel_ptx)
        launch_kernel(ptx_path, (1, 1, 1), (64, 1, 1), self.ema_buffer, 256)

    def set_mode(self, mode: int):
        if mode not in [0, 1, 2]:
            raise ValueError("Invalid thinking tag mode")
        mode_val = ctypes.c_uint32(mode)
        memcpy_htod(self.mode_buffer, ctypes.byref(mode_val), 4)

    def _get_mode(self) -> int:
        mode = ctypes.c_uint32()
        memcpy_dtoh(ctypes.byref(mode), self.mode_buffer, 4)
        return mode.value

    def _warm_resonance_cache(self):
        """Grok's Resonance Cache Warmer"""
        logger.info("Warming resonance cache...")
        hot_queries = self._get_hot_freq_regions()
        for query in hot_queries:
            try:
                self.resonance_field.query(query, preload_cache=True)
            except Exception as e:
                logger.warning(f"Failed to preload cache: {e}")

    def _get_hot_freq_regions(self):
        """Identify frequently accessed regions"""
        # Placeholder - would query Galaxy for high access_freq embeddings
        return [np.random.randn(512).astype(np.float32) for _ in range(10)]

    def inference(self, input_embedding: np.ndarray, modal_signature: list, temporal_anchor: float = None):
        mode = self._get_mode()
        self.latency_guard.start()
        try:
            # 1. Adaptive sparsity
            sparsity_level = self.adaptive_sparsity.calculate_sparsity(
                input_embedding, modal_signature
            )

            # 2. Fetch weight trajectories
            trajectories = self.resonance_field.query(
                input_embedding, 
                sparsity=sparsity_level,
                time_window=temporal_anchor or 0.5
            )

            # 3. Cross-modal resonance patterns
            enriched_embeddings = self.cross_modal_engine.apply_resonance_pattern(
                trajectories, modal_signature
            )

            # 4. Sparse weight assembly
            sparse_weights = self.adaptive_sparsity.apply_adaptive_sparsity(
                enriched_embeddings, sparsity_level
            )

            # 5. Temporal coherence
            context = self.temporal_reasoning.compute_deltas(enriched_embeddings)

            # 6. RPN program execution
            if mode == self.MODE_FULL_TEMPORAL:
                output = self._execute_temporal_mlp(
                    input_embedding, sparse_weights, context
                )
            elif mode == self.MODE_SPATIAL_ONLY:
                output = self._execute_spatial_mlp(
                    input_embedding, sparse_weights
                )
            else:
                output = self._execute_temporal_mlp(
                    input_embedding, sparse_weights, context
                )

            # 7. Graph crystallization
            crystallized = self.graph_crystallizer.apply(output, self.ema_buffer)

            # 8. Vector resonance for confidence
            confidence_rays = self.vector_resonator.compute(confidence_vector=crystallized)

            # 9. Entropy and uncertainty
            uncertainty = self._compute_entropy(crystallized)
            coherence_scores = self.temporal_reasoning.estimate_coherence(context)

            # 10. Optional visualization
            if self.visualizer and mode == 2:
                self.visualizer.visualize_inference_flow(input_embedding, tags)

            output_obj = ThinkingTagOutput(crystallized, confidence_rays, uncertainty, coherence_scores)
            tags = []
            if uncertainty > 0.5:
                tags.append(("uncertainty", uncertainty))
            output_obj.tags = tags
            return output_obj

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._recover_fallback(input_embedding, modal_signature, error=e)
        finally:
            self.latency_guard.stop()

    def _build_temporal_rpn_program(self, weights, context):
        """Exact RPN bytecode builder (Kimi's fix)"""
        from .modular_rpn_engine import RPNProgram, OP_SPARSE_LOAD, OP_SMAV, OP_ENTROPY_SUM
        
        p = RPNProgram()
        # Layer 1
        p.u32(OP_SPARSE_LOAD)
        p.ptr(weights['W1'])
        p.u32(OP_SMAV)
        p.f32(0.0)
        p.u8(0x0A)  # MAX
        
        # Temporal gate
        p.u8(0xF0)  # CALL temporal_coherence
        p.ptr(context)
        p.u8(0x12)  # MUL
        
        # Layer 2
        p.u32(OP_SPARSE_LOAD)
        p.ptr(weights['W2'])
        p.u32(OP_SMAV)
        p.u8(0xF1)  # CALL temporal_mask
        p.u8(0x12)  # MUL
        p.f32(0.0)
        p.u8(0x0A)  # MAX
        
        # Dynamic crystallize
        p.u8(0xF2)  # CALL crystallize_intermediate
        p.ptr(self.ema_buffer)
        
        # Layer 3
        p.u32(OP_SPARSE_LOAD)
        p.ptr(weights['W3'])
        p.u32(OP_SMAV)
        p.u8(0x0B)  # SIGMOID_APPROX
        
        # Entropy
        p.u8(0x06)  # DUP
        p.u32(OP_ENTROPY_SUM)
        
        return p

    def _execute_temporal_mlp(self, x, weights, context):
        program = self._build_temporal_rpn_program(weights, context)
        return self.rpn_engine.eval(program, [x])

    def _build_spatial_rpn_program(self, weights):
        from .modular_rpn_engine import RPNProgram, OP_SPARSE_LOAD, OP_SMAV
        
        p = RPNProgram()
        for layer_key in ['W1', 'W2', 'W3']:
            w = weights[layer_key]
            p.u32(OP_SPARSE_LOAD)
            p.ptr(w)
            p.u32(OP_SMAV)
            if layer_key != 'W3':
                p.f32(0.0)
                p.u8(0x0A)  # MAX
        return p

    def _execute_spatial_mlp(self, x, weights):
        if self._cached_spatial_prog is None:
            self._cached_spatial_prog = self._build_spatial_rpn_program(weights)
        return self.rpn_engine.eval(self._cached_spatial_prog, [x])

    def _recover_fallback(self, input_emb, modal_sig, error=None):
        """Sovereign fallback without recursion (Kimi's fix)"""
        if error:
            logger.warning(f"Fallback after {type(error).__name__}")
        
        # Direct spatial path execution
        trajectories = self.resonance_field.query(
            input_emb, sparsity=0.1, region="thinking_weights"
        )
        sparse_weights = self._assemble_sparse_weights(trajectories)
        
        output = self._execute_spatial_mlp(input_emb, sparse_weights)
        probs = self._sigmoid_approx(output)
        
        # Simple fallback output
        tags = [("uncertainty", 0.99)]  # Signal fallback occurred
        return ThinkingTagOutput(
            probs, 
            np.ones_like(probs), 
            0.99, 
            np.zeros_like(probs)
        )

    def _compute_entropy(self, probs):
        clipped = np.clip(probs, 1e-6, 1.0)
        return float(-np.sum(clipped * np.log(clipped)))

    def _sigmoid_approx(self, x):
        return 0.5 * (1.0 + np.tanh(0.5 * x))

    def _detect_error_trajectories(self, data):
        if isinstance(data, np.ndarray):
            return np.any(np.isnan(data)) or np.any(np.isinf(data))
        return False

    def _assemble_sparse_weights(self, trajectories):
        # Placeholder - would use AtomicFissionFusion for real assembly
        return {
            'W1': np.random.randn(256, 512).astype(np.float32),
            'W2': np.random.randn(256, 256).astype(np.float32),
            'W3': np.random.randn(100, 256).astype(np.float32)
        }

    def _extract_temporal_context(self, trajectories):
        return np.random.randn(256).astype(np.float32)

    def _get_house_priors(self):
        return np.random.randn(100, 256).astype(np.float32)
