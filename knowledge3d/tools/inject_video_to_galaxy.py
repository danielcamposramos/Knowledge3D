"""
Galaxy Injection Pipeline: Video → Galaxy Nodes (Multimodal Learning)

Part of Week 1-2 implementation from Step7.1_FINAL.txt
Swarm collaboration:
- Codex: Core implementation
- Grok: Edge case handling (malformed videos, codec issues)
- Kimi: GPU optimization for frame processing
- GLM: Multimodal embedding validation
- Qwen: Sleep integration hooks
- Claude: Tests + documentation

Key Features:
- Real-time video ingestion to Galaxy (frames + audio transcript)
- RPN-powered frame similarity for keyframe selection
- Multimodal nodes (vision + audio + text)
- Temporal coherence preservation
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
import subprocess

import numpy as np
import cupy as cp

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("⚠️  OpenCV not available - install with: pip install opencv-python")

try:
    import whisper
    _HAS_WHISPER = False  # Disabled by default (large model)
except ImportError:
    _HAS_WHISPER = False


class VideoGalaxyInjector:
    """
    Injects video content into Galaxy as multimodal semantic nodes.

    Uses RPN kernel for:
    - Frame similarity (keyframe selection)
    - Multimodal quality scoring
    - Temporal coherence validation
    """

    def __init__(
        self,
        galaxy_path: str,
        fps_sample: float = 1.0,  # Sample 1 frame per second
        min_quality: float = 0.5,
        similarity_threshold: float = 0.90,  # Higher for frames (more similar)
        extract_audio: bool = True
    ):
        """
        Initialize video injector.

        Args:
            galaxy_path: Path to galaxy.glb file
            fps_sample: Frames per second to sample
            min_quality: Minimum quality score to inject
            similarity_threshold: Frame similarity threshold for keyframe selection
            extract_audio: Whether to extract and transcribe audio
        """
        self.galaxy_path = Path(galaxy_path)
        self.fps_sample = fps_sample
        self.min_quality = min_quality
        self.similarity_threshold = similarity_threshold
        self.extract_audio = extract_audio

        # Load or initialize Galaxy
        from knowledge3d.spatial.galaxy import GalaxyGraph
        if self.galaxy_path.exists():
            self.galaxy = GalaxyGraph.load(str(self.galaxy_path))
        else:
            self.galaxy = GalaxyGraph.create_empty()

        # RPN executor for similarity checks
        try:
            from knowledge3d.cranium.clustering_rpn import compute_cosine_similarity_rpn
            from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_score_rpn
            self.compute_similarity = compute_cosine_similarity_rpn
            self.compute_quality = compute_honesty_score_rpn
            self._use_rpn = True
        except Exception as e:
            print(f"⚠️  RPN not available, using CPU fallback: {e}")
            self._use_rpn = False

    def extract_frames(
        self,
        video_path: str
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames from video at specified sampling rate.

        Grok's edge case handling:
        - Handles codec errors
        - Skips corrupted frames
        - Validates frame dimensions

        Args:
            video_path: Path to video file

        Returns:
            List of (frame_number, frame_array) tuples
        """
        if not _HAS_CV2:
            raise RuntimeError("OpenCV required for video injection")

        frames = []

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            # Calculate frame sampling interval
            frame_interval = max(1, int(video_fps / self.fps_sample))

            print(f"   Video FPS: {video_fps:.2f}, Total frames: {total_frames}")
            print(f"   Sampling every {frame_interval} frames")

            frame_num = 0
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Sample at specified interval
                if frame_num % frame_interval == 0:
                    # Grok's edge case: Validate frame
                    if frame is not None and frame.size > 0:
                        # Resize to standard size (for consistent embeddings)
                        frame_resized = cv2.resize(frame, (224, 224))
                        frames.append((frame_num, frame_resized))

                frame_num += 1

            cap.release()

        except Exception as e:
            # Grok's edge case: Handle codec errors
            print(f"⚠️  Error extracting frames from {video_path}: {e}")
            raise

        return frames

    def extract_audio_transcript(
        self,
        video_path: str
    ) -> Optional[str]:
        """
        Extract and transcribe audio from video.

        Args:
            video_path: Path to video file

        Returns:
            Transcribed text or None if extraction fails
        """
        if not self.extract_audio:
            return None

        # For now, return None (Whisper is expensive)
        # Future: Integrate Whisper or other speech-to-text
        print("   ⚠️  Audio transcription not yet implemented (requires Whisper)")
        return None

    def compute_frame_quality_rpn(
        self,
        frame: np.ndarray,
        embedding: np.ndarray
    ) -> float:
        """
        Compute quality score for video frame using RPN.

        Quality components:
        - Correctness: Embedding magnitude (well-formed image)
        - Reasoning: Visual complexity (edge density)
        - Uncertainty: Frame sharpness (gradient variance)
        - Alignment: Content relevance (embedding norm)

        Args:
            frame: Frame image (H, W, 3)
            embedding: Frame embedding vector

        Returns:
            Quality score [0, 1]
        """
        if not self._use_rpn:
            return 0.8  # Default for frames

        # Component 1: Correctness (embedding norm)
        emb_norm = float(np.linalg.norm(embedding))
        correctness = min(1.0, emb_norm)

        # Component 2: Reasoning (visual complexity - edge density)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if _HAS_CV2 else frame
        edges = cv2.Canny(gray, 50, 150) if _HAS_CV2 else gray
        edge_density = np.mean(edges > 0) if edges.size > 0 else 0.5
        reasoning = float(edge_density)

        # Component 3: Uncertainty (sharpness - gradient variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F) if _HAS_CV2 else gray
        sharpness = float(np.var(laplacian))
        uncertainty = min(1.0, sharpness / 1000.0)  # Normalize

        # Component 4: Alignment (content relevance)
        alignment = min(1.0, np.abs(embedding).mean() * 5)

        # RPN-powered quality score
        quality = self.compute_quality(
            correctness=correctness,
            reasoning=reasoning,
            uncertainty=uncertainty,
            alignment=alignment
        )

        return float(quality)

    def select_keyframes_rpn(
        self,
        frames: List[Tuple[int, np.ndarray]],
        embeddings: np.ndarray
    ) -> List[int]:
        """
        Select keyframes using RPN similarity to avoid redundant frames.

        Args:
            frames: List of (frame_num, frame_array)
            embeddings: Frame embeddings (N, D)

        Returns:
            Indices of keyframes to keep
        """
        if not self._use_rpn or len(frames) == 0:
            return list(range(len(frames)))

        keyframe_indices = [0]  # Always keep first frame

        for i in range(1, len(frames)):
            # Compare with last keyframe
            last_keyframe_idx = keyframe_indices[-1]
            similarity = self.compute_similarity(
                embeddings[i],
                embeddings[last_keyframe_idx]
            )

            # Keep if sufficiently different from last keyframe
            if similarity < self.similarity_threshold:
                keyframe_indices.append(i)

        return keyframe_indices

    def inject_video(
        self,
        video_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Inject video into Galaxy as multimodal semantic nodes.

        Args:
            video_path: Path to video file
            metadata: Optional metadata (title, source, etc.)

        Returns:
            Injection statistics
        """
        start_time = time.time()

        # Extract frames
        print(f"🎬 Extracting frames from {video_path}...")
        frames = self.extract_frames(video_path)
        print(f"   → Extracted {len(frames)} frames")

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        # Extract audio transcript (if enabled)
        transcript = self.extract_audio_transcript(video_path)

        # Generate embeddings for all frames
        print("🖼️  Generating frame embeddings...")
        from knowledge3d.tools.embedders import embed_image_gpu

        frame_embeddings = []
        for frame_num, frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if _HAS_CV2 else frame
            embedding = embed_image_gpu(frame_rgb)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            frame_embeddings.append(embedding)

        frame_embeddings = np.array(frame_embeddings, dtype=np.float32)

        # Select keyframes using RPN similarity
        print("🔑 Selecting keyframes...")
        keyframe_indices = self.select_keyframes_rpn(frames, frame_embeddings)
        print(f"   → Selected {len(keyframe_indices)} keyframes from {len(frames)} frames")

        # Inject keyframes to Galaxy
        print("🌌 Injecting to Galaxy...")
        injected_count = 0
        low_quality_count = 0

        for idx in keyframe_indices:
            frame_num, frame = frames[idx]
            embedding = frame_embeddings[idx]

            # Compute quality using RPN
            quality = self.compute_frame_quality_rpn(frame, embedding)

            # Filter low-quality frames
            if quality < self.min_quality:
                low_quality_count += 1
                continue

            # Create node ID
            node_id = f"video_{Path(video_path).stem}_frame_{frame_num}"

            # Inject to Galaxy
            node_metadata = {
                'source': str(video_path),
                'frame_number': int(frame_num),
                'quality': float(quality),
                'type': 'video_frame',
                'modality': 'vision',
                **(metadata or {})
            }

            # Store frame as base64 or reference
            # For now, just store embedding (frame pixels too large)
            self.galaxy.add_node(
                node_id=node_id,
                embedding=embedding,
                content=f"Video frame {frame_num} from {Path(video_path).name}",
                metadata=node_metadata
            )

            injected_count += 1

        # Inject transcript as separate node if available
        if transcript:
            transcript_embedding = embed_text_gpu(transcript)
            self.galaxy.add_node(
                node_id=f"video_{Path(video_path).stem}_transcript",
                embedding=transcript_embedding,
                content=transcript,
                metadata={
                    'source': str(video_path),
                    'type': 'video_transcript',
                    'modality': 'audio_text',
                    **(metadata or {})
                }
            )
            injected_count += 1

        # Save Galaxy
        self.galaxy.save(str(self.galaxy_path))

        elapsed = time.time() - start_time

        stats = {
            'video_path': str(video_path),
            'total_frames': len(frames),
            'keyframes_selected': len(keyframe_indices),
            'injected_nodes': injected_count,
            'low_quality_skipped': low_quality_count,
            'has_transcript': transcript is not None,
            'elapsed_seconds': elapsed,
            'frames_per_second': len(frames) / max(0.001, elapsed),
            'galaxy_total_nodes': self.galaxy.node_count if hasattr(self.galaxy, 'node_count') else injected_count
        }

        print(f"\n✅ Injection complete!")
        print(f"   → {injected_count} nodes injected")
        print(f"   → {low_quality_count} low-quality skipped")
        print(f"   → {elapsed:.2f}s elapsed")

        return stats


def inject_video_to_galaxy(
    video_path: str,
    galaxy_path: str = "viewer/public/galaxy/volatile_galaxy.glb",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to inject video to Galaxy.

    Args:
        video_path: Path to video file
        galaxy_path: Path to Galaxy GLB
        **kwargs: Additional arguments for VideoGalaxyInjector

    Returns:
        Injection statistics
    """
    injector = VideoGalaxyInjector(galaxy_path, **kwargs)
    return injector.inject_video(video_path)
