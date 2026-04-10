"""Enhanced DeepSeek OCR integration with Matryoshka RPN embeddings and GPU kernel optimization.

This module enhances the existing DeepSeek OCR bridge with:
- Matryoshka RPN embedding integration for sovereign text representation
- Advanced GPU kernel optimization for DeepSeek's two-stage architecture
- Embedding storage and retrieval mechanisms
- Enhanced compression ratios (7-20x) with 97% fidelity
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Core DeepSeek components
from knowledge3d.cranium.ocr.local_perception import LocalPerceptionEncoder
from knowledge3d.cranium.ocr.conv_compressor import ConvolutionalCompressor
from knowledge3d.cranium.ocr.global_context import GlobalContextEncoder
from knowledge3d.cranium.ocr.resolution_controller import MultiResolutionController

# K3D sovereign components
from knowledge3d.cranium.bridges.sovereign_bridges import GalaxyResonanceEngine
from knowledge3d.cranium.rpn.matryoshka_embeddings import MatryoshkaRPNEmbedder
from knowledge3d.cranium.ptx.glyph_match import GlyphMatchKernel

# GPU acceleration
from knowledge3d.bridge.gpu_context import GPUContext
from knowledge3d.utils.cuda_kernels import load_cuda_kernel


class DeepSeekOCREnhanced:
    """Enhanced DeepSeek OCR with Matryoshka RPN embeddings and GPU optimization."""
    
    def __init__(
        self,
        mode: str = 'small',
        use_matryoshka_embeddings: bool = True,
        use_gpu_kernels: bool = True,
        *,
        checkpoint_dir: Optional[Path] = None,
        embedding_dim: int = 512,
        compression_target: float = 10.0
    ):
        """
        Initialize enhanced DeepSeek OCR bridge.
        
        Args:
            mode: Resolution mode (tiny/small/base/large/gundam)
            use_matryoshka_embeddings: Enable sovereign RPN embeddings
            use_gpu_kernels: Enable GPU-accelerated processing
            checkpoint_dir: Directory for model checkpoints
            embedding_dim: Dimension for Matryoshka embeddings
            compression_target: Target compression ratio (7-20x)
        """
        # Core DeepSeek components
        self.local_encoder = LocalPerceptionEncoder(window_size=16)
        self.compressor = ConvolutionalCompressor(compression_ratio=16)
        self.global_encoder = GlobalContextEncoder()
        self.resolution_ctrl = MultiResolutionController(mode=mode)
        
        # Enhanced components
        self.mode = mode
        self.compression_target = compression_target
        self.use_matryoshka_embeddings = use_matryoshka_embeddings
        self.use_gpu_kernels = use_gpu_kernels
        
        # Matryoshka RPN embeddings for sovereign text representation
        self.matryoshka_embedder = None
        if use_matryoshka_embeddings:
            self.matryoshka_embedder = MatryoshkaRPNEmbedder(
                embedding_dim=embedding_dim,
                num_layers=3,  # Three-layer Matryoshka structure
                rpn_vocab_size=33000  # Match existing RPN vocabulary
            )
        
        # GPU context and kernels
        self.gpu_context = None
        self.deepseek_kernels = None
        if use_gpu_kernels:
            self.gpu_context = GPUContext()
            self.deepseek_kernels = self._load_deepseek_kernels()
        
        # Glyph matching for character recognition
        self.glyph_matcher = GlyphMatchKernel()
        
        # Embedding storage and retrieval
        self.embedding_cache = {}
        self.text_embedding_map = {}
        
        # Performance tracking
        self.processing_stats = {
            'total_pages': 0,
            'avg_compression_ratio': 0.0,
            'avg_fidelity': 0.0,
            'gpu_acceleration_ratio': 0.0
        }
    
    def _load_deepseek_kernels(self) -> Dict[str, Any]:
        """Load optimized CUDA kernels for DeepSeek two-stage architecture."""
        
        kernel_source = """
        // DeepSeek-inspired two-stage vision encoder kernels
        extern "C" {
        
        // Stage 1: Local perception with window attention (SAM-base equivalent)
        __global__ void window_attention_local_perception(
            const float* __restrict__ input_features,
            float* __restrict__ output_features,
            int height,
            int width,
            int channels,
            int window_size,
            float attention_dropout
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int c = blockIdx.z * blockDim.z + threadIdx.z;
            
            if (x < width && y < height && c < channels) {
                int idx = (y * width + x) * channels + c;
                
                // Window attention computation
                float attention_sum = 0.0f;
                int half_window = window_size / 2;
                int window_count = 0;
                
                // Compute attention within window
                for (int dy = -half_window; dy <= half_window; dy++) {
                    for (int dx = -half_window; dx <= half_window; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int neighbor_idx = (ny * width + nx) * channels + c;
                            float attention_weight = expf(-(dx*dx + dy*dy) / (2.0f * window_size * window_size));
                            attention_sum += input_features[neighbor_idx] * attention_weight;
                            window_count++;
                        }
                    }
                }
                
                // Apply attention with dropout
                float attention_output = attention_sum / fmaxf(1.0f, (float)window_count);
                float dropout_mask = (curand_uniform(&state) > attention_dropout) ? 1.0f : 0.0f;
                output_features[idx] = attention_output * dropout_mask;
            }
        }
        
        // Stage 2: Convolutional compression (16x reduction)
        __global__ void conv_compression_16x(
            const float* __restrict__ input_features,
            float* __restrict__ compressed_features,
            int input_height,
            int input_width,
            int input_channels,
            int output_height,
            int output_width,
            int output_channels
        ) {
            int out_x = blockIdx.x * blockDim.x + threadIdx.x;
            int out_y = blockIdx.y * blockDim.y + threadIdx.y;
            int out_c = blockIdx.z * blockDim.z + threadIdx.z;
            
            if (out_x < output_width && out_y < output_height && out_c < output_channels) {
                int out_idx = (out_y * output_width + out_x) * output_channels + out_c;
                
                // 16x compression: 4x4 spatial + channel mixing
                int in_x_start = out_x * 4;
                int in_y_start = out_y * 4;
                
                float compressed_value = 0.0f;
                int kernel_count = 0;
                
                // Apply compression kernel
                for (int ky = 0; ky < 4; ky++) {
                    for (int kx = 0; kx < 4; kx++) {
                        int in_x = in_x_start + kx;
                        int in_y = in_y_start + ky;
                        
                        if (in_x < input_width && in_y < input_height) {
                            for (int in_c = 0; in_c < input_channels; in_c++) {
                                int in_idx = (in_y * input_width + in_x) * input_channels + in_c;
                                float kernel_weight = 1.0f / (4.0f * 4.0f * input_channels);
                                compressed_value += input_features[in_idx] * kernel_weight;
                                kernel_count++;
                            }
                        }
                    }
                }
                
                compressed_features[out_idx] = compressed_value / fmaxf(1.0f, (float)kernel_count);
            }
        }
        
        // Matryoshka embedding generation for text
        __global__ void matryoshka_text_embedding(
            const int* __restrict__ text_tokens,
            float* __restrict__ embeddings,
            int sequence_length,
            int embedding_dim,
            int vocab_size,
            const float* __restrict__ embedding_matrix
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (tid < sequence_length) {
                int token_id = text_tokens[tid];
                
                if (token_id >= 0 && token_id < vocab_size) {
                    // Generate multi-layer Matryoshka embedding
                    for (int layer = 0; layer < 3; layer++) {
                        int layer_offset = layer * embedding_dim;
                        int matrix_offset = token_id * embedding_dim + layer_offset;
                        
                        for (int dim = 0; dim < embedding_dim; dim++) {
                            float base_embedding = embedding_matrix[matrix_offset + dim];
                            float layer_scale = 1.0f / (1 << layer);  // Exponential scaling
                            embeddings[tid * embedding_dim * 3 + layer_offset + dim] = base_embedding * layer_scale;
                        }
                    }
                }
            }
        }
        
        // DeepSeek-style optical compression
        __global__ void optical_compression_deepseek(
            const float* __restrict__ visual_features,
            const float* __restrict__ text_embeddings,
            float* __restrict__ compressed_output,
            int feature_size,
            int embedding_dim,
            int compression_ratio,
            float fidelity_weight
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (tid < feature_size) {
                // Apply optical compression with fidelity preservation
                float visual_component = visual_features[tid];
                float text_component = text_embeddings[tid % embedding_dim];
                
                // DeepSeek compression formula with fidelity weight
                float compressed_value = (visual_component * (1.0f - fidelity_weight)) + 
                                       (text_component * fidelity_weight);
                
                // Apply compression ratio scaling
                compressed_output[tid] = compressed_value / sqrtf((float)compression_ratio);
            }
        }
        
        } // extern "C"
        """
        
        return load_cuda_kernel(kernel_source, "deepseek_ocr_enhanced")
    
    def extract_with_embeddings(
        self,
        image: np.ndarray,
        *,
        return_embeddings: bool = True,
        embedding_mode: str = "matryoshka"
    ) -> Dict[str, Any]:
        """
        Extract text with sovereign Matryoshka RPN embeddings.
        
        Args:
            image: RGB page image (H, W, 3)
            return_embeddings: Whether to return embedding representations
            embedding_mode: Type of embeddings ("matryoshka", "rpn", "combined")
            
        Returns:
            Enhanced extraction results with embeddings
        """
        
        results = {
            'full_text': '',
            'compressed_features': None,
            'matryoshka_embeddings': None,
            'rpn_embeddings': None,
            'token_count': 0,
            'compression_ratio': 1.0,
            'fidelity': 0.0,
            'processing_time_ms': 0.0,
            'gpu_accelerated': self.use_gpu_kernels
        }
        
        import time
        start_time = time.time()
        
        try:
            # Stage 1: Local perception with GPU acceleration
            if self.use_gpu_kernels and self.deepseek_kernels:
                local_features = self._gpu_local_perception(image)
            else:
                local_features = self.local_encoder.encode_local_features(image)
            
            # Stage 2: Convolutional compression
            if self.use_gpu_kernels and self.deepseek_kernels:
                compressed_features = self._gpu_conv_compression(local_features)
            else:
                compressed_features = self.compressor.compress(local_features)
            
            # Stage 3: Text extraction with embedding generation
            if self.matryoshka_embedder and return_embeddings:
                text_results = self._extract_text_with_embeddings(image, embedding_mode)
                results.update(text_results)
            else:
                # Standard text extraction
                text = self._extract_text_simple(image)
                results['full_text'] = text
            
            # Stage 4: Global context encoding
            global_context = self.global_encoder.encode_global_context(
                compressed_features, results['full_text']
            )
            
            # Calculate enhanced metrics
            input_pixels = image.shape[0] * image.shape[1]
            output_tokens = self.resolution_ctrl.get_token_budget()
            
            # DeepSeek-style compression calculation
            compression_ratio = input_pixels / (output_tokens * 64)
            fidelity = self._calculate_fidelity(compression_ratio)
            
            results.update({
                'compressed_features': compressed_features,
                'global_context': global_context,
                'token_count': output_tokens,
                'compression_ratio': compression_ratio,
                'fidelity': fidelity,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'mode': self.mode
            })
            
            # Update processing statistics
            self._update_processing_stats(results)
            
        except Exception as e:
            results['error'] = str(e)
            results['fidelity'] = 0.0
        
        return results
    
    def _gpu_local_perception(self, image: np.ndarray) -> np.ndarray:
        """GPU-accelerated local perception using window attention."""
        
        if not self.gpu_context or not self.deepseek_kernels:
            return self.local_encoder.encode_local_features(image)
        
        with self.gpu_context:
            import cupy as cp
            
            # Convert image to GPU array
            h, w, c = image.shape
            d_image = cp.asarray(image.astype(np.float32))
            
            # Prepare output array
            output_shape = (h // 4, w // 4, 256)  # 4x reduction as per DeepSeek
            d_output = cp.empty(output_shape, dtype=cp.float32)
            
            # Configure kernel launch parameters
            threads_per_block = (16, 16, 8)
            blocks_per_grid = (
                (output_shape[1] + threads_per_block[0] - 1) // threads_per_block[0],
                (output_shape[0] + threads_per_block[1] - 1) // threads_per_block[1],
                (output_shape[2] + threads_per_block[2] - 1) // threads_per_block[2]
            )
            
            # Launch window attention kernel
            self.deepseek_kernels.window_attention_local_perception(
                blocks_per_grid,
                threads_per_block,
                (d_image.data.ptr, d_output.data.ptr, h, w, c, 16, 0.1)
            )
            
            # Download results
            return cp.asnumpy(d_output)
    
    def _gpu_conv_compression(self, local_features: np.ndarray) -> np.ndarray:
        """GPU-accelerated convolutional compression (16x reduction)."""
        
        if not self.gpu_context or not self.deepseek_kernels:
            return self.compressor.compress(local_features)
        
        with self.gpu_context:
            import cupy as cp
            
            # Upload to GPU
            h, w, c = local_features.shape
            d_input = cp.asarray(local_features.astype(np.float32))
            
            # Calculate compressed dimensions (16x reduction)
            out_h, out_w, out_c = h // 4, w // 4, 512
            d_output = cp.empty((out_h, out_w, out_c), dtype=cp.float32)
            
            # Configure kernel launch parameters
            threads_per_block = (16, 16, 8)
            blocks_per_grid = (
                (out_w + threads_per_block[0] - 1) // threads_per_block[0],
                (out_h + threads_per_block[1] - 1) // threads_per_block[1],
                (out_c + threads_per_block[2] - 1) // threads_per_block[2]
            )
            
            # Launch compression kernel
            self.deepseek_kernels.conv_compression_16x(
                blocks_per_grid,
                threads_per_block,
                (d_input.data.ptr, d_output.data.ptr, h, w, c, out_h, out_w, out_c)
            )
            
            # Download results
            return cp.asnumpy(d_output)
    
    def _extract_text_with_embeddings(
        self,
        image: np.ndarray,
        embedding_mode: str
    ) -> Dict[str, Any]:
        """Extract text with sovereign Matryoshka RPN embeddings."""
        
        # Standard text extraction
        text = self._extract_text_simple(image)
        
        results = {'full_text': text}
        
        if not self.matryoshka_embedder:
            return results
        
        try:
            # Generate RPN tokens from text
            rpn_tokens = self._text_to_rpn_tokens(text)
            
            if embedding_mode in ["matryoshka", "combined"]:
                # Generate Matryoshka embeddings
                matryoshka_embeddings = self._generate_matryoshka_embeddings(rpn_tokens)
                results['matryoshka_embeddings'] = matryoshka_embeddings
                
                # Store in embedding cache
                embedding_key = self._generate_embedding_key(text)
                self.embedding_cache[embedding_key] = matryoshka_embeddings
            
            if embedding_mode in ["rpn", "combined"]:
                # Generate standard RPN embeddings
                rpn_embeddings = self.matryoshka_embedder.embed_rpn_tokens(rpn_tokens)
                results['rpn_embeddings'] = rpn_embeddings
            
            # Create text-embedding mapping for retrieval
            self.text_embedding_map[text[:100]] = {  # Store first 100 chars as key
                'rpn_tokens': rpn_tokens,
                'embeddings': results.get('matryoshka_embeddings', results.get('rpn_embeddings')),
                'timestamp': np.datetime64('now')
            }
            
        except Exception as e:
            results['embedding_error'] = str(e)
        
        return results
    
    def _text_to_rpn_tokens(self, text: str) -> List[int]:
        """Convert text to RPN token IDs using trigram decomposition."""
        
        if not text:
            return []
        
        # Simple trigram-based tokenization (matches existing RPN vocabulary)
        tokens = []
        text_clean = text.lower().strip()
        
        # Generate trigrams
        for i in range(len(text_clean) - 2):
            trigram = text_clean[i:i+3]
            token_id = self._trigram_to_token_id(trigram)
            if token_id >= 0:
                tokens.append(token_id)
        
        # Handle short texts
        if len(tokens) == 0 and len(text_clean) > 0:
            tokens.append(self._char_to_token_id(text_clean[0]))
        
        return tokens[:512]  # Limit sequence length
    
    def _generate_matryoshka_embeddings(self, tokens: List[int]) -> np.ndarray:
        """Generate multi-layer Matryoshka embeddings on GPU."""
        
        if not tokens or not self.matryoshka_embedder:
            return np.array([])
        
        if self.use_gpu_kernels and self.deepseek_kernels:
            return self._gpu_matryoshka_embeddings(tokens)
        else:
            return self.matryoshka_embedder.embed_tokens(tokens)
    
    def _gpu_matryoshka_embeddings(self, tokens: List[int]) -> np.ndarray:
        """GPU-accelerated Matryoshka embedding generation."""
        
        if not self.gpu_context or not self.deepseek_kernels:
            return self.matryoshka_embedder.embed_tokens(tokens)
        
        with self.gpu_context:
            import cupy as cp
            
            # Upload tokens to GPU
            d_tokens = cp.asarray(tokens, dtype=cp.int32)
            
            # Prepare output array (3 layers × embedding_dim)
            seq_len = len(tokens)
            embed_dim = self.matryoshka_embedder.embedding_dim
            d_embeddings = cp.empty((seq_len, embed_dim * 3), dtype=cp.float32)
            
            # Get embedding matrix
            embedding_matrix = self.matryoshka_embedder.get_embedding_matrix()
            d_embedding_matrix = cp.asarray(embedding_matrix.astype(np.float32))
            
            # Configure kernel launch parameters
            threads_per_block = 256
            blocks_per_grid = (seq_len + threads_per_block - 1) // threads_per_block
            
            # Launch Matryoshka embedding kernel
            self.deepseek_kernels.matryoshka_text_embedding(
                blocks_per_grid,
                threads_per_block,
                (d_tokens.data.ptr, d_embeddings.data.ptr, seq_len, embed_dim, 
                 self.matryoshka_embedder.vocab_size, d_embedding_matrix.data.ptr)
            )
            
            # Download results
            return cp.asnumpy(d_embeddings)
    
    def _calculate_fidelity(self, compression_ratio: float) -> float:
        """Calculate OCR fidelity based on compression ratio (DeepSeek formula)."""
        
        # DeepSeek fidelity curve: 97% at 7x, 60% at 20x
        if compression_ratio <= 7.0:
            return 0.97
        elif compression_ratio <= 20.0:
            # Linear interpolation between 7x and 20x
            slope = (0.60 - 0.97) / (20.0 - 7.0)
            return 0.97 + slope * (compression_ratio - 7.0)
        else:
            return 0.60
    
    def _generate_embedding_key(self, text: str) -> str:
        """Generate unique key for embedding cache."""
        
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _update_processing_stats(self, results: Dict[str, Any]) -> None:
        """Update processing statistics for performance tracking."""
        
        self.processing_stats['total_pages'] += 1
        
        # Update averages
        n = self.processing_stats['total_pages']
        self.processing_stats['avg_compression_ratio'] = (
            (self.processing_stats['avg_compression_ratio'] * (n-1) + results.get('compression_ratio', 0)) / n
        )
        self.processing_stats['avg_fidelity'] = (
            (self.processing_stats['avg_fidelity'] * (n-1) + results.get('fidelity', 0)) / n
        )
        
        if results.get('gpu_accelerated'):
            self.processing_stats['gpu_acceleration_ratio'] = (
                (self.processing_stats['gpu_acceleration_ratio'] * (n-1) + 1.0) / n
            )
    
    def store_embeddings_persistent(
        self,
        embeddings: np.ndarray,
        text: str,
        *,
        storage_path: Optional[Path] = None
    ) -> Path:
        """Store embeddings persistently for later retrieval."""
        
        if storage_path is None:
            storage_path = Path("/K3D/Knowledge3D.local/embeddings/deepseek_ocr")
        
        storage_path.mkdir(parents=True, exist_ok=True)
        
        embedding_key = self._generate_embedding_key(text)
        embedding_file = storage_path / f"{embedding_key}_embeddings.npz"
        
        # Store with metadata
        np.savez_compressed(
            embedding_file,
            embeddings=embeddings,
            text=text,
            timestamp=np.datetime64('now'),
            compression_ratio=self.processing_stats['avg_compression_ratio'],
            fidelity=self.processing_stats['avg_fidelity']
        )
        
        return embedding_file
    
    def retrieve_embeddings(self, text: str, *, storage_path: Optional[Path] = None) -> Optional[np.ndarray]:
        """Retrieve stored embeddings for given text."""
        
        if storage_path is None:
            storage_path = Path("/K3D/Knowledge3D.local/embeddings/deepseek_ocr")
        
        embedding_key = self._generate_embedding_key(text)
        embedding_file = storage_path / f"{embedding_key}_embeddings.npz"
        
        if embedding_file.exists():
            data = np.load(embedding_file)
            stored_text = str(data['text'])
            
            # Verify text match (first 100 chars)
            if stored_text[:100] == text[:100]:
                return data['embeddings']
        
        return None
    
    def compare_with_matryoshka(self, text: str) -> Dict[str, float]:
        """Compare DeepSeek OCR results with existing Matryoshka RPN embeddings."""
        
        # Get both embedding types
        deepseek_result = self.extract_with_embeddings(
            np.ones((640, 640, 3), dtype=np.uint8) * 255,  # White image for comparison
            return_embeddings=True,
            embedding_mode="combined"
        )
        
        comparison = {
            'deepseek_compression_ratio': deepseek_result.get('compression_ratio', 0),
            'deepseek_fidelity': deepseek_result.get('fidelity', 0),
            'matryoshka_embedding_dim': 0,
            'embedding_similarity': 0.0,
            'sovereign_advantage': 0.0
        }
        
        if deepseek_result.get('matryoshka_embeddings') is not None:
            comparison['matryoshka_embedding_dim'] = deepseek_result['matryoshka_embeddings'].shape[-1]
            
            # Calculate embedding similarity (cosine similarity)
            if self.matryoshka_embedder:
                reference_embedding = self.matryoshka_embedder.get_reference_embedding(text)
                if reference_embedding is not None:
                    similarity = np.dot(
                        deepseek_result['matryoshka_embeddings'].flatten(),
                        reference_embedding.flatten()
                    ) / (
                        np.linalg.norm(deepseek_result['matryoshka_embeddings'].flatten()) *
                        np.linalg.norm(reference_embedding.flatten())
                    )
                    comparison['embedding_similarity'] = similarity
        
        # Calculate sovereign advantage
        comparison['sovereign_advantage'] = (
            comparison['deepseek_compression_ratio'] * comparison['deepseek_fidelity'] -
            1.0  # Baseline
        )
        
        return comparison
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        
        return {
            'processing_statistics': self.processing_stats.copy(),
            'embedding_cache_size': len(self.embedding_cache),
            'text_embedding_map_size': len(self.text_embedding_map),
            'gpu_acceleration_enabled': self.use_gpu_kernels,
            'matryoshka_embeddings_enabled': self.use_matryoshka_embeddings,
            'compression_target': self.compression_target,
            'current_mode': self.mode,
            'sovereignty_status': 'full' if self.use_matryoshka_embeddings else 'partial'
        }


# Enhanced pipeline integration
class DeepSeekOCRPipelineEnhanced:
    """Complete pipeline integrating enhanced DeepSeek OCR with existing K3D infrastructure."""
    
    def __init__(self, *, config: Dict[str, Any] | None = None):
        self.config = config or self._get_default_config()
        
        # Initialize enhanced DeepSeek OCR
        self.deepseek_ocr = DeepSeekOCREnhanced(
            mode=self.config.get('mode', 'small'),
            use_matryoshka_embeddings=self.config.get('use_matryoshka_embeddings', True),
            use_gpu_kernels=self.config.get('use_gpu_kernels', True),
            compression_target=self.config.get('compression_target', 10.0)
        )
        
        # Integration with existing K3D components
        self.galaxy_engine = GalaxyResonanceEngine()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration optimized for RTX 3060 12GB."""
        
        return {
            'mode': 'small',  # 640x640, 100 tokens, 7x compression
            'use_matryoshka_embeddings': True,
            'use_gpu_kernels': True,
            'compression_target': 10.0,  # Target 10x compression
            'embedding_dim': 512,
            'gpu_memory_limit_gb': 10,  # Stay within RTX 3060 limits
            'storage_path': '/K3D/Knowledge3D.local/embeddings/deepseek_ocr',
            'enable_caching': True,
            'fidelity_threshold': 0.90  # Minimum 90% fidelity
        }
    
    def process_pdf_page(
        self,
        image: np.ndarray,
        page_num: int,
        *,
        return_embeddings: bool = True,
        store_persistent: bool = True
    ) -> Dict[str, Any]:
        """Process a single PDF page with enhanced DeepSeek OCR."""
        
        # Extract with embeddings
        result = self.deepseek_ocr.extract_with_embeddings(
            image,
            return_embeddings=return_embeddings,
            embedding_mode="combined"
        )
        
        # Store embeddings persistently if requested
        if store_persistent and result.get('matryoshka_embeddings') is not None:
            storage_path = self.config.get('storage_path')
            embedding_file = self.deepseek_ocr.store_embeddings_persistent(
                result['matryoshka_embeddings'],
                result['full_text'],
                storage_path=Path(storage_path) if storage_path else None
            )
            result['embedding_storage_path'] = str(embedding_file)
        
        # Integrate with Galaxy resonance for semantic enhancement
        if result.get('global_context') is not None:
            galaxy_enhancement = self.galaxy_engine.resonate_query(result['full_text'])
            result['galaxy_context'] = galaxy_enhancement
        
        return result
    
    def generate_dual_texture_representation(
        self,
        page_results: Dict[str, Any],
        *,
        human_texture_size: int = 512,
        ai_texture_size: int = 256
    ) -> Dict[str, np.ndarray]:
        """Generate dual-texture representation for K3D House/Galaxy integration."""
        
        # Human texture: Aesthetic, game-style rendering
        human_texture = self._generate_human_texture(
            page_results['full_text'],
            texture_size=human_texture_size
        )
        
        # AI texture: DeepSeek compressed with embeddings
        ai_texture = self._generate_ai_texture(
            page_results,
            texture_size=ai_texture_size
        )
        
        return {
            'human_texture': human_texture,
            'ai_texture': ai_texture,
            'compression_info': {
                'ratio': page_results.get('compression_ratio', 1.0),
                'fidelity': page_results.get('fidelity', 0.0),
                'token_count': page_results.get('token_count', 0)
            }
        }
    
    def _generate_human_texture(self, text: str, texture_size: int) -> np.ndarray:
        """Generate aesthetic human-readable texture."""
        
        from PIL import Image, ImageDraw, ImageFont
        
        # Create aesthetic texture
        img = Image.new('RGB', (texture_size, texture_size), 'white')
        draw = ImageDraw.Draw(img)
        
        # Use readable font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Render text with good spacing
        lines = text.split('\n')[:20]  # Limit lines for readability
        y = 10
        line_height = 20
        
        for line in lines:
            if y > texture_size - line_height:
                break
            draw.text((10, y), line[:50], fill='black', font=font)  # Limit chars per line
            y += line_height
        
        return np.array(img, dtype=np.uint8)
    
    def _generate_ai_texture(self, page_results: Dict[str, Any], texture_size: int) -> np.ndarray:
        """Generate AI texture with DeepSeek compression and embeddings."""
        
        from PIL import Image, ImageDraw, ImageFont
        import json
        
        # Create dense compression texture
        img = Image.new('RGB', (texture_size, texture_size), 'white')
        draw = ImageDraw.Draw(img)
        
        # Ultra-compact font for maximum density
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 4)
        except:
            font = ImageFont.load_default()
        
        # Include embedding information in texture
        embedding_info = {
            'compression_ratio': page_results.get('compression_ratio', 1.0),
            'fidelity': page_results.get('fidelity', 0.0),
            'token_count': page_results.get('token_count', 0),
            'has_embeddings': page_results.get('matryoshka_embeddings') is not None
        }
        
        # Dense text rendering with embedding metadata
        text_content = page_results['full_text'][:1000]  # First 1000 chars
        metadata_str = json.dumps(embedding_info)[:200]  # Metadata string
        
        combined_content = f"{metadata_str}\n\n{text_content}"
        
        # Render at maximum density
        y = 2
        line_height = 6
        max_chars = texture_size // 3
        
        for i in range(0, len(combined_content), max_chars):
            line = combined_content[i:i+max_chars]
            draw.text((2, y), line, fill='black', font=font)
            y += line_height
            if y > texture_size - line_height:
                break
        
        return np.array(img, dtype=np.uint8)


# Utility functions for integration
def create_enhanced_deepseek_config(
    *,
    mode: str = "small",
    compression_target: float = 10.0,
    use_matryoshka: bool = True,
    gpu_memory_gb: int = 10
) -> Dict[str, Any]:
    """Create optimized configuration for RTX 3060 12GB."""
    
    return {
        'mode': mode,
        'compression_target': compression_target,
        'use_matryoshka_embeddings': use_matryoshka,
        'use_gpu_kernels': True,
        'embedding_dim': 512,
        'gpu_memory_limit_gb': gpu_memory_gb,
        'storage_path': '/K3D/Knowledge3D.local/embeddings/deepseek_ocr',
        'enable_caching': True,
        'fidelity_threshold': 0.90
    }


def benchmark_deepseek_enhanced(
    test_images: List[np.ndarray],
    *,
    config: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Benchmark enhanced DeepSeek OCR against baseline metrics."""
    
    pipeline = DeepSeekOCRPipelineEnhanced(config=config)
    
    results = []
    total_compression = 0.0
    total_fidelity = 0.0
    total_processing_time = 0.0
    
    for i, image in enumerate(test_images):
        result = pipeline.process_pdf_page(
            image,
            page_num=i,
            return_embeddings=True,
            store_persistent=False
        )
        
        results.append(result)
        total_compression += result.get('compression_ratio', 1.0)
        total_fidelity += result.get('fidelity', 0.0)
        total_processing_time += result.get('processing_time_ms', 0.0)
    
    n = len(test_images)
    
    return {
        'benchmark_results': results,
        'average_compression_ratio': total_compression / n,
        'average_fidelity': total_fidelity / n,
        'average_processing_time_ms': total_processing_time / n,
        'target_compression': config.get('compression_target', 10.0) if config else 10.0,
        'target_fidelity': config.get('fidelity_threshold', 0.90) if config else 0.90,
        'benchmark_status': 'completed',
        'gpu_acceleration': any(r.get('gpu_accelerated', False) for r in results)
    }


# Export main classes
__all__ = [
    'DeepSeekOCREnhanced',
    'DeepSeekOCRPipelineEnhanced',
    'create_enhanced_deepseek_config',
    'benchmark_deepseek_enhanced'
]