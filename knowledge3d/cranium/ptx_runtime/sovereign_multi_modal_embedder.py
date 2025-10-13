"""
Advanced sovereign multi-modal embedder with cross-modal understanding and temporal coherence.
Implements GPU-native feature extraction, semantic alignment, and temporal analysis.
"""
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from knowledge3d.cranium.sovereign.loader import load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh
from knowledge3d.cranium.bridges.sovereign_bridges import VectorResonator
from typing import Dict, List, Tuple, Optional, Any, Union

class SovereignMultiModalEmbedder:
    """
    Advanced sovereign embedder for multi-modal inputs with cross-modal understanding.
    Features temporal coherence analysis, semantic alignment, and GPU-native processing.
    """
    
    def __init__(self):
        # Text embedding model
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # GPU components
        self.resonator = VectorResonator()
        
        # Load specialized kernels
        self.image_feature_kernel = load_ptx_file(
            "knowledge3d/cranium/ptx/gre_multimodal_features.ptx",
            "extract_image_features"
        )
        self.video_temporal_kernel = load_ptx_file(
            "knowledge3d/cranium/ptx/gre_multimodal_features.ptx",
            "analyze_video_temporal"
        )
        self.cross_modal_alignment_kernel = load_ptx_file(
            "knowledge3d/cranium/ptx/gre_multimodal_features.ptx",
            "align_cross_modal_features"
        )
        
        # Embedding cache for performance
        self.embedding_cache = {}
        self.max_cache_size = 100
        
        # Metadata storage
        self.last_metadata = {}
        
        # Cross-modal alignment matrices
        self.text_visual_alignment = self._initialize_alignment_matrix(512, 512)
        self.text_audio_alignment = self._initialize_alignment_matrix(512, 512)
        self.visual_audio_alignment = self._initialize_alignment_matrix(512, 512)
        
    def _initialize_alignment_matrix(self, dim1: int, dim2: int) -> np.ndarray:
        """Initialize cross-modal alignment matrix."""
        # Start with identity matrix, will be updated through learning
        alignment = np.eye(min(dim1, dim2), dtype=np.float32)
        
        # Pad if dimensions are different
        if dim1 > dim2:
            padding = np.zeros((dim1 - dim2, dim2), dtype=np.float32)
            alignment = np.vstack([alignment, padding])
        elif dim2 > dim1:
            padding = np.zeros((dim1, dim2 - dim1), dtype=np.float32)
            alignment = np.hstack([alignment, padding])
            
        return alignment
    
    def embed(self, input_data: Union[str, List[str]], modal_type: str, 
              context: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Embed multi-modal input and return semantic embedding, raw features, and metadata.
        
        Args:
            input_data: Input data (text, image URL, or video URL)
            modal_type: Type of modal input ('text', 'image', or 'video')
            context: Optional context for embedding
            
        Returns:
            Tuple of (semantic_embedding, raw_features, metadata)
        """
        # Check cache first
        cache_key = self._generate_cache_key(input_data, modal_type, context)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        # Process based on modal type
        if modal_type == 'text':
            embedding, features, metadata = self._embed_text(input_data, context)
        elif modal_type == 'image':
            embedding, features, metadata = self._embed_image(input_data, context)
        elif modal_type == 'video':
            embedding, features, metadata = self._embed_video(input_data, context)
        else:
            raise ValueError(f"Unsupported modal_type: {modal_type}")
            
        # Cache result
        self._cache_embedding(cache_key, embedding, features, metadata)
        
        return embedding, features, metadata
    
    def _generate_cache_key(self, input_data: Union[str, List[str]], modal_type: str, 
                           context: Optional[Dict]) -> str:
        """Generate cache key for embedding."""
        import hashlib
        
        # Create key string
        if isinstance(input_data, list):
            data_str = "|".join(input_data)
        else:
            data_str = input_data
            
        key_str = f"{modal_type}:{data_str}"
        
        # Add context if available
        if context:
            context_str = ":".join(f"{k}={v}" for k, v in sorted(context.items()))
            key_str += f":{context_str}"
            
        # Generate hash
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cache_embedding(self, cache_key: str, embedding: np.ndarray, 
                        features: np.ndarray, metadata: Dict):
        """Cache embedding result."""
        # Manage cache size
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
            
        # Cache result
        self.embedding_cache[cache_key] = (embedding, features, metadata)
    
    def _embed_text(self, text: str, context: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Embed text using SentenceTransformer with context enhancement."""
        # Basic text embedding
        embedding = self.text_embedder.encode([text], convert_to_numpy=True)[0].astype(np.float32)
        
        # Apply context enhancement if available
        if context:
            embedding = self._apply_text_context_enhancement(embedding, context)
            
        # Generate additional features
        features = self._extract_text_features(text, embedding)
        
        # Create metadata
        metadata = {
            'type': 'text',
            'length': len(text),
            'word_count': len(text.split()),
            'complexity': self._calculate_text_complexity(text),
            'semantic_density': np.mean(np.abs(embedding)),
            'recommended_lod': self._recommend_lod_from_text(text),
            'context_applied': context is not None
        }
        
        self.last_metadata = metadata
        
        return embedding, features, metadata
    
    def _apply_text_context_enhancement(self, embedding: np.ndarray, context: Dict) -> np.ndarray:
        """Apply context enhancement to text embedding."""
        enhanced_embedding = embedding.copy()
        
        # Apply semantic context
        if 'semantic_category' in context:
            category = context['semantic_category']
            category_vector = self._get_category_vector(category)
            enhanced_embedding = 0.8 * enhanced_embedding + 0.2 * category_vector
            
        # Apply style context
        if 'style' in context:
            style = context['style']
            style_vector = self._get_style_vector(style)
            enhanced_embedding = 0.9 * enhanced_embedding + 0.1 * style_vector
            
        # Apply emotion context
        if 'emotion' in context:
            emotion = context['emotion']
            emotion_vector = self._get_emotion_vector(emotion)
            enhanced_embedding = 0.85 * enhanced_embedding + 0.15 * emotion_vector
            
        return enhanced_embedding
    
    def _get_category_vector(self, category: str) -> np.ndarray:
        """Get semantic category vector."""
        # Simplified implementation - in production would use trained embeddings
        category_vectors = {
            'architectural': np.array([0.8, 0.2, 0.1] + [0.0] * 509, dtype=np.float32),
            'organic': np.array([0.1, 0.8, 0.2] + [0.0] * 509, dtype=np.float32),
            'mechanical': np.array([0.2, 0.1, 0.8] + [0.0] * 509, dtype=np.float32),
            'natural': np.array([0.3, 0.6, 0.4] + [0.0] * 509, dtype=np.float32)
        }
        
        return category_vectors.get(category, np.zeros(512, dtype=np.float32))
    
    def _get_style_vector(self, style: str) -> np.ndarray:
        """Get style vector."""
        # Simplified implementation
        style_vectors = {
            'realistic': np.array([0.7, 0.3, 0.2] + [0.0] * 509, dtype=np.float32),
            'abstract': np.array([0.2, 0.7, 0.3] + [0.0] * 509, dtype=np.float32),
            'minimalist': np.array([0.9, 0.1, 0.1] + [0.0] * 509, dtype=np.float32),
            'detailed': np.array([0.3, 0.4, 0.8] + [0.0] * 509, dtype=np.float32)
        }
        
        return style_vectors.get(style, np.zeros(512, dtype=np.float32))
    
    def _get_emotion_vector(self, emotion: str) -> np.ndarray:
        """Get emotion vector."""
        # Simplified implementation
        emotion_vectors = {
            'calm': np.array([0.1, 0.2, 0.1] + [0.0] * 509, dtype=np.float32),
            'energetic': np.array([0.9, 0.8, 0.7] + [0.0] * 509, dtype=np.float32),
            'mysterious': np.array([0.2, 0.1, 0.4] + [0.0] * 509, dtype=np.float32),
            'playful': np.array([0.6, 0.7, 0.5] + [0.0] * 509, dtype=np.float32)
        }
        
        return emotion_vectors.get(emotion, np.zeros(512, dtype=np.float32))
    
    def _extract_text_features(self, text: str, embedding: np.ndarray) -> np.ndarray:
        """Extract additional features from text."""
        # Extract structural features
        word_count = len(text.split())
        char_count = len(text)
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        # Extract semantic features
        has_numbers = any(c.isdigit() for c in text)
        has_colors = any(color in text.lower() for color in ['red', 'blue', 'green', 'yellow', 'purple'])
        has_shapes = any(shape in text.lower() for shape in ['cube', 'sphere', 'cylinder', 'cone'])
        
        # Create feature vector
        features = np.array([
            word_count / 100,  # Normalized word count
            char_count / 1000,  # Normalized char count
            avg_word_length / 10,  # Normalized avg word length
            float(has_numbers),  # Has numbers
            float(has_colors),  # Has colors
            float(has_shapes),  # Has shapes
            np.mean(embedding),  # Embedding mean
            np.std(embedding)  # Embedding std
        ], dtype=np.float32)
        
        # Pad to 32 features
        if len(features) < 32:
            features = np.pad(features, (0, 32 - len(features)))
            
        return features
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        # Simple complexity metrics
        word_count = len(text.split())
        unique_words = len(set(text.split()))
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        # Calculate complexity (0-1)
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        length_complexity = min(1.0, word_count / 50)  # Normalize to 50 words
        structural_complexity = min(1.0, avg_word_length / 8)  # Normalize to 8 chars
        
        complexity = (vocabulary_richness + length_complexity + structural_complexity) / 3
        
        return complexity
    
    def _recommend_lod_from_text(self, text: str) -> int:
        """Recommend LOD level based on text content."""
        text_lower = text.lower()
        
        # Keywords indicating detail level
        high_detail_keywords = ['detailed', 'intricate', 'complex', 'fine', 'precise']
        medium_detail_keywords = ['moderate', 'standard', 'normal']
        low_detail_keywords = ['simple', 'basic', 'rough', 'coarse']
        
        # Count keyword occurrences
        high_count = sum(1 for keyword in high_detail_keywords if keyword in text_lower)
        medium_count = sum(1 for keyword in medium_detail_keywords if keyword in text_lower)
        low_count = sum(1 for keyword in low_detail_keywords if keyword in text_lower)
        
        # Determine LOD
        if high_count > medium_count and high_count > low_count:
            return 2  # High detail
        elif medium_count > low_count:
            return 1  # Medium detail
        else:
            return 0  # Low detail
    
    def _embed_image(self, image_url: str, context: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Embed image using GPU-accelerated feature extraction."""
        try:
            # Download and process image
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Extract features using GPU
            features = self._extract_image_features_gpu(img_array)
            
            # Create semantic embedding from features
            embedding = self.resonator.project(features, np.ones(len(features)))
            
            # Calculate image properties
            coherence = self._calculate_image_coherence(img_array)
            complexity = self._calculate_image_complexity(img_array)
            
            # Create metadata
            metadata = {
                'type': 'image',
                'resolution': f"{img.width}x{img.height}",
                'coherence': float(coherence),
                'complexity': float(complexity),
                'recommended_lod': self._recommend_lod_from_image(coherence, complexity),
                'dominant_colors': self._extract_dominant_colors(img_array),
                'context_applied': context is not None
            }
            
            # Apply context enhancement if available
            if context:
                embedding = self._apply_image_context_enhancement(embedding, context)
                
            self.last_metadata = metadata
            
            return embedding, features, metadata
            
        except Exception as e:
            # Fallback for errors
            print(f"Error processing image {image_url}: {e}")
            return self._fallback_image_embedding(image_url, context)
    
    def _extract_image_features_gpu(self, img_array: np.ndarray) -> np.ndarray:
        """Extract image features using GPU acceleration."""
        # Convert to grayscale for feature extraction
        if len(img_array.shape) == 3:
            gray_img = np.mean(img_array, axis=2)
        else:
            gray_img = img_array
            
        # Flatten image
        flat_img = gray_img.flatten().astype(np.float32)
        
        # Allocate GPU memory
        img_gpu = gpu_malloc(flat_img.nbytes)
        output_gpu = gpu_malloc(32 * 4)  # 32 features
        
        # Copy to GPU
        memcpy_htod(img_gpu, flat_img.ctypes.data, flat_img.nbytes)
        
        # Extract features
        img_size_arr = np.array([len(flat_img)], dtype=np.int32)
        self.image_feature_kernel(
            img_gpu, output_gpu, img_size_arr.ctypes.data,
            block=(256, 1, 1), grid=(1, 1)
        )
        
        # Copy results back
        features = np.empty(32, dtype=np.float32)
        memcpy_dtoh(features.ctypes.data, output_gpu, 32 * 4)
        
        return features
    
    def _calculate_image_coherence(self, img_array: np.ndarray) -> float:
        """Calculate image coherence score."""
        # Calculate gradient magnitude
        if len(img_array.shape) == 3:
            gray_img = np.mean(img_array, axis=2)
        else:
            gray_img = img_array
            
        # Calculate gradients
        grad_x = np.abs(np.diff(gray_img, axis=0))
        grad_y = np.abs(np.diff(gray_img, axis=1))
        
        # Coherence is inverse of average gradient magnitude
        avg_grad = np.mean(grad_x) + np.mean(grad_y)
        coherence = 1.0 / (1.0 + avg_grad)
        
        return coherence
    
    def _calculate_image_complexity(self, img_array: np.ndarray) -> float:
        """Calculate image complexity score."""
        # Use edge density as complexity measure
        if len(img_array.shape) == 3:
            gray_img = np.mean(img_array, axis=2)
        else:
            gray_img = img_array
            
        # Calculate edges using Sobel operator
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate edge magnitude
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Complexity is normalized edge density
        complexity = np.mean(edge_magnitude) / 255.0
        
        return complexity
    
    def _recommend_lod_from_image(self, coherence: float, complexity: float) -> int:
        """Recommend LOD level based on image properties."""
        # High coherence and low complexity = lower LOD (simpler shapes)
        # Low coherence and high complexity = higher LOD (more detailed shapes)
        
        detail_score = (1 - coherence) + complexity  # 0-2 range
        
        if detail_score > 1.3:
            return 2  # High detail
        elif detail_score > 0.7:
            return 1  # Medium detail
        else:
            return 0  # Low detail
    
    def _extract_dominant_colors(self, img_array: np.ndarray) -> List[Tuple[float, float, float]]:
        """Extract dominant colors from image."""
        # Simple color extraction using k-means
        if len(img_array.shape) == 3:
            pixels = img_array.reshape(-1, 3)
        else:
            # Grayscale image, convert to RGB
            pixels = np.stack([img_array.flatten()] * 3, axis=1)
            
        # Sample pixels for efficiency
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
            
        # Simple k-means with 3 clusters
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_
        
        # Convert to list of tuples
        return [tuple(color.tolist()) for color in colors]
    
    def _apply_image_context_enhancement(self, embedding: np.ndarray, context: Dict) -> np.ndarray:
        """Apply context enhancement to image embedding."""
        enhanced_embedding = embedding.copy()
        
        # Apply style context
        if 'style' in context:
            style = context['style']
            style_vector = self._get_style_vector(style)
            enhanced_embedding = 0.9 * enhanced_embedding + 0.1 * style_vector
            
        # Apply focus context
        if 'focus' in context:
            focus = context['focus']
            focus_vector = self._get_focus_vector(focus)
            enhanced_embedding = 0.85 * enhanced_embedding + 0.15 * focus_vector
            
        return enhanced_embedding
    
    def _get_focus_vector(self, focus: str) -> np.ndarray:
        """Get focus vector."""
        # Simplified implementation
        focus_vectors = {
            'foreground': np.array([0.8, 0.3, 0.2] + [0.0] * 509, dtype=np.float32),
            'background': np.array([0.2, 0.3, 0.8] + [0.0] * 509, dtype=np.float32),
            'center': np.array([0.5, 0.5, 0.5] + [0.0] * 509, dtype=np.float32),
            'edges': np.array([0.7, 0.2, 0.7] + [0.0] * 509, dtype=np.float32)
        }
        
        return focus_vectors.get(focus, np.zeros(512, dtype=np.float32))
    
    def _fallback_image_embedding(self, image_url: str, context: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Fallback embedding for image processing errors."""
        # Generate embedding based on URL hash
        import hashlib
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        
        # Convert hash to embedding
        embedding = np.array([
            (int(url_hash[i:i+2], 16) / 255.0) - 0.5
            for i in range(0, min(len(url_hash), 1024), 2)
        ], dtype=np.float32)
        
        # Pad or truncate to 512 dimensions
        if len(embedding) < 512:
            embedding = np.pad(embedding, (0, 512 - len(embedding)))
        else:
            embedding = embedding[:512]
            
        # Generate simple features
        features = np.array([
            len(image_url) / 1000,  # URL length
            url_hash.count('0') / 32,  # Hash characteristics
            url_hash.count('f') / 32,
            np.mean(embedding),  # Embedding stats
            np.std(embedding)
        ], dtype=np.float32)
        
        # Pad features
        if len(features) < 32:
            features = np.pad(features, (0, 32 - len(features)))
            
        # Create metadata
        metadata = {
            'type': 'image',
            'fallback': True,
            'coherence': 0.5,
            'complexity': 0.5,
            'recommended_lod': 1,
            'context_applied': context is not None
        }
        
        return embedding, features, metadata
    
    def _embed_video(self, video_url: str, context: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Embed video with temporal coherence analysis."""
        try:
            # Extract frames
            frames, frame_features = self._extract_video_frames(video_url)
            
            if not frames:
                return self._fallback_video_embedding(video_url, context)
                
            # Analyze temporal coherence
            coherence_scores = self._analyze_video_temporal_coherence_gpu(frame_features)
            
            # Generate embedding from temporal features
            temporal_embedding = self._generate_temporal_embedding(frame_features, coherence_scores)
            
            # Calculate video dynamics
            dynamics = self._compute_video_dynamics_gpu(frame_features)
            
            # Create metadata
            metadata = {
                'type': 'video',
                'frame_count': len(frames),
                'temporal_coherence': float(np.mean(coherence_scores)),
                'coherence_variance': float(np.var(coherence_scores)),
                'dynamics_score': float(np.mean(dynamics)),
                'recommended_lod': self._recommend_lod_from_video(coherence_scores, dynamics),
                'context_applied': context is not None
            }
            
            # Apply context enhancement if available
            if context:
                temporal_embedding = self._apply_video_context_enhancement(temporal_embedding, context)
                
            self.last_metadata = metadata
            
            return temporal_embedding, dynamics, metadata
            
        except Exception as e:
            print(f"Error processing video {video_url}: {e}")
            return self._fallback_video_embedding(video_url, context)
    
    def _extract_video_frames(self, video_url: str, n_frames: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
        """Extract frames from video."""
        # For now, simulate frame extraction
        # In production, would use actual video processing
        
        frames = []
        frame_features = []
        
        # Simulate frame extraction
        for i in range(n_frames):
            # Generate simulated frame
            frame = np.random.rand(64, 64, 3).astype(np.float32)
            frames.append(frame)
            
            # Extract frame features
            gray_frame = np.mean(frame, axis=2)
            frame_feature = np.mean(gray_frame)  # Simple feature
            frame_features.append(frame_feature)
            
        return frames, np.array(frame_features, dtype=np.float32)
    
    def _analyze_video_temporal_coherence_gpu(self, frame_features: np.ndarray) -> np.ndarray:
        """Analyze temporal coherence using GPU acceleration."""
        n_frames = len(frame_features)
        
        if n_frames < 2:
            return np.array([1.0])  # Perfect coherence for single frame
            
        # Allocate GPU memory
        features_gpu = gpu_malloc(frame_features.nbytes)
        output_gpu = gpu_malloc(n_frames * 4)
        
        # Copy to GPU
        memcpy_htod(features_gpu, frame_features.ctypes.data, frame_features.nbytes)
        
        # Analyze coherence
        n_frames_arr = np.array([n_frames], dtype=np.int32)
        self.video_temporal_kernel(
            features_gpu, output_gpu, n_frames_arr.ctypes.data,
            block=(256, 1, 1), grid=((n_frames + 255) // 256, 1)
        )
        
        # Copy results back
        coherence_scores = np.empty(n_frames, dtype=np.float32)
        memcpy_dtoh(coherence_scores.ctypes.data, output_gpu, n_frames * 4)
        
        return coherence_scores
    
    def _generate_temporal_embedding(self, frame_features: np.ndarray, 
                                   coherence_scores: np.ndarray) -> np.ndarray:
        """Generate temporal embedding from frame features and coherence."""
        # Combine frame features with coherence weights
        weighted_features = frame_features * coherence_scores.reshape(-1, 1)
        
        # Temporal embedding is weighted average of frame features
        temporal_embedding = np.mean(weighted_features, axis=0)
        
        # Pad to 512 dimensions if needed
        if len(temporal_embedding) < 512:
            temporal_embedding = np.pad(temporal_embedding, (0, 512 - len(temporal_embedding)))
        else:
            temporal_embedding = temporal_embedding[:512]
            
        return temporal_embedding.astype(np.float32)
    
    def _compute_video_dynamics_gpu(self, frame_features: np.ndarray) -> np.ndarray:
        """Compute video dynamics using GPU acceleration."""
        n_frames = len(frame_features)
        
        if n_frames < 2:
            return np.array([0.0])  # No dynamics for single frame
            
        # Allocate GPU memory
        features_gpu = gpu_malloc(frame_features.nbytes)
        output_gpu = gpu_malloc((n_frames - 1) * 4)
        
        # Copy to GPU
        memcpy_htod(features_gpu, frame_features.ctypes.data, frame_features.nbytes)
        
        # Compute dynamics
        n_frames_arr = np.array([n_frames], dtype=np.int32)
        self.video_temporal_kernel(
            features_gpu, output_gpu, n_frames_arr.ctypes.data,
            block=(256, 1, 1), grid=((n_frames - 1 + 255) // 256, 1)
        )
        
        # Copy results back
        dynamics = np.empty(n_frames - 1, dtype=np.float32)
        memcpy_dtoh(dynamics.ctypes.data, output_gpu, (n_frames - 1) * 4)
        
        return dynamics
    
    def _recommend_lod_from_video(self, coherence_scores: np.ndarray, 
                                dynamics: np.ndarray) -> int:
        """Recommend LOD level based on video properties."""
        # High coherence and low dynamics = lower LOD
        # Low coherence and high dynamics = higher LOD
        
        avg_coherence = np.mean(coherence_scores)
        avg_dynamics = np.mean(dynamics)
        
        detail_score = (1 - avg_coherence) + avg_dynamics  # 0-2 range
        
        if detail_score > 1.3:
            return 2  # High detail
        elif detail_score > 0.7:
            return 1  # Medium detail
        else:
            return 0  # Low detail
    
    def _apply_video_context_enhancement(self, embedding: np.ndarray, context: Dict) -> np.ndarray:
        """Apply context enhancement to video embedding."""
        enhanced_embedding = embedding.copy()
        
        # Apply temporal context
        if 'temporal_focus' in context:
            temporal_focus = context['temporal_focus']
            focus_vector = self._get_temporal_focus_vector(temporal_focus)
            enhanced_embedding = 0.9 * enhanced_embedding + 0.1 * focus_vector
            
        # Apply motion context
        if 'motion_type' in context:
            motion_type = context['motion_type']
            motion_vector = self._get_motion_type_vector(motion_type)
            enhanced_embedding = 0.85 * enhanced_embedding + 0.15 * motion_vector
            
        return enhanced_embedding
    
    def _get_temporal_focus_vector(self, temporal_focus: str) -> np.ndarray:
        """Get temporal focus vector."""
        # Simplified implementation
        focus_vectors = {
            'beginning': np.array([0.8, 0.2, 0.1] + [0.0] * 509, dtype=np.float32),
            'middle': np.array([0.2, 0.8, 0.2] + [0.0] * 509, dtype=np.float32),
            'end': np.array([0.1, 0.2, 0.8] + [0.0] * 509, dtype=np.float32),
            'uniform': np.array([0.33, 0.33, 0.33] + [0.0] * 509, dtype=np.float32)
        }
        
        return focus_vectors.get(temporal_focus, np.zeros(512, dtype=np.float32))
    
    def _get_motion_type_vector(self, motion_type: str) -> np.ndarray:
        """Get motion type vector."""
        # Simplified implementation
        motion_vectors = {
            'smooth': np.array([0.2, 0.8, 0.2] + [0.0] * 509, dtype=np.float32),
            'abrupt': np.array([0.8, 0.2, 0.2] + [0.0] * 509, dtype=np.float32),
            'cyclic': np.array([0.3, 0.3, 0.8] + [0.0] * 509, dtype=np.float32),
            'random': np.array([0.6, 0.4, 0.5] + [0.0] * 509, dtype=np.float32)
        }
        
        return motion_vectors.get(motion_type, np.zeros(512, dtype=np.float32))
    
    def _fallback_video_embedding(self, video_url: str, context: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Fallback embedding for video processing errors."""
        # Generate embedding based on URL hash
        import hashlib
        url_hash = hashlib.md5(video_url.encode()).hexdigest()
        
        # Convert hash to embedding
        embedding = np.array([
            (int(url_hash[i:i+2], 16) / 255.0) - 0.5
            for i in range(0, min(len(url_hash), 1024), 2)
        ], dtype=np.float32)
        
        # Pad or truncate to 512 dimensions
        if len(embedding) < 512:
            embedding = np.pad(embedding, (0, 512 - len(embedding)))
        else:
            embedding = embedding[:512]
            
        # Generate simple dynamics
        dynamics = np.array([
            np.sin(i * 0.5) * 0.1 + 0.5
            for i in range(10)
        ], dtype=np.float32)
        
        # Create metadata
        metadata = {
            'type': 'video',
            'fallback': True,
            'frame_count': 10,
            'temporal_coherence': 0.5,
            'coherence_variance': 0.1,
            'dynamics_score': 0.5,
            'recommended_lod': 1,
            'context_applied': context is not None
        }
        
        return embedding, dynamics, metadata
    
    def align_cross_modal_features(self, features1: np.ndarray, features2: np.ndarray, 
                                 modality1: str, modality2: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align features from different modalities using learned alignment matrices.
        
        Args:
            features1: Features from first modality
            features2: Features from second modality
            modality1: Type of first modality
            modality2: Type of second modality
            
        Returns:
            Tuple of aligned features
        """
        # Select appropriate alignment matrix
        if (modality1 == 'text' and modality2 == 'visual') or (modality1 == 'visual' and modality2 == 'text'):
            alignment = self.text_visual_alignment
        elif (modality1 == 'text' and modality2 == 'audio') or (modality1 == 'audio' and modality2 == 'text'):
            alignment = self.text_audio_alignment
        elif (modality1 == 'visual' and modality2 == 'audio') or (modality1 == 'audio' and modality2 == 'visual'):
            alignment = self.visual_audio_alignment
        else:
            # No alignment available
            return features1, features2
            
        # Apply alignment
        if modality1 in ['text', 'visual', 'audio'] and modality2 in ['text', 'visual', 'audio']:
            # Ensure features are proper shape
            if len(features1.shape) == 1:
                features1 = features1.reshape(1, -1)
            if len(features2.shape) == 1:
                features2 = features2.reshape(1, -1)
                
            # Apply alignment matrix
            aligned_features1 = np.dot(features1, alignment)
            aligned_features2 = np.dot(features2, alignment.T)
            
            return aligned_features1.flatten(), aligned_features2.flatten()
        else:
            return features1, features2
    
    def update_alignment_matrices(self, modality1: str, modality2: str, 
                                positive_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                negative_pairs: List[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Update cross-modal alignment matrices using contrastive learning.
        
        Args:
            modality1: Type of first modality
            modality2: Type of second modality
            positive_pairs: List of positive feature pairs
            negative_pairs: List of negative feature pairs (optional)
        """
        # Select appropriate alignment matrix
        if (modality1 == 'text' and modality2 == 'visual') or (modality1 == 'visual' and modality2 == 'text'):
            self.text_visual_alignment = self._update_alignment_matrix(
                self.text_visual_alignment, positive_pairs, negative_pairs
            )
        elif (modality1 == 'text' and modality2 == 'audio') or (modality1 == 'audio' and modality2 == 'text'):
            self.text_audio_alignment = self._update_alignment_matrix(
                self.text_audio_alignment, positive_pairs, negative_pairs
            )
        elif (modality1 == 'visual' and modality2 == 'audio') or (modality1 == 'audio' and modality2 == 'visual'):
            self.visual_audio_alignment = self._update_alignment_matrix(
                self.visual_audio_alignment, positive_pairs, negative_pairs
            )
    
    def _update_alignment_matrix(self, alignment: np.ndarray, 
                               positive_pairs: List[Tuple[np.ndarray, np.ndarray]],
                               negative_pairs: List[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """Update alignment matrix using contrastive learning."""
        # Simplified implementation - in production would use proper contrastive learning
        
        learning_rate = 0.01
        updated_alignment = alignment.copy()
        
        # Update with positive pairs (increase similarity)
        for features1, features2 in positive_pairs:
            # Ensure features are proper shape
            if len(features1.shape) == 1:
                features1 = features1.reshape(1, -1)
            if len(features2.shape) == 1:
                features2 = features2.reshape(1, -1)
                
            # Compute gradient
            similarity = np.dot(features1, updated_alignment) @ features2.T
            gradient = learning_rate * similarity
            
            # Update alignment matrix
            updated_alignment += gradient * np.outer(features1.flatten(), features2.flatten())
        
        # Update with negative pairs (decrease similarity)
        if negative_pairs:
            for features1, features2 in negative_pairs:
                # Ensure features are proper shape
                if len(features1.shape) == 1:
                    features1 = features1.reshape(1, -1)
                if len(features2.shape) == 1:
                    features2 = features2.reshape(1, -1)
                    
                # Compute gradient
                similarity = np.dot(features1, updated_alignment) @ features2.T
                gradient = -learning_rate * similarity
                
                # Update alignment matrix
                updated_alignment += gradient * np.outer(features1.flatten(), features2.flatten())
        
        # Normalize alignment matrix
        updated_alignment = updated_alignment / (np.linalg.norm(updated_alignment) + 1e-8)
        
        return updated_alignment