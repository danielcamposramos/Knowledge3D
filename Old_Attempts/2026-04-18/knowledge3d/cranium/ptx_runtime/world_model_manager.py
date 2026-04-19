"""
World Model Manager - GLM's Multi-Modal Temporal System
Manages world state, temporal coherence, and dynamic mesh generation.
"""
import numpy as np
from knowledge3d.cranium.bridges.sovereign_bridges import WorldModelBridge


class WorldModelManager:
    """
    Manages world model state and operations for multi-modal 3D generation.

    Features:
    - Temporal coherence analysis across video frames
    - Multi-modal feature fusion (text + visual)
    - World state prediction and evolution
    - Dynamic mesh generation based on world state
    - Galaxy resonance enhancement
    """

    def __init__(self, max_history=10):
        self.bridge = WorldModelBridge()
        self.state_history = []
        self.max_history = max_history

    def compute_video_coherence(self, frame_features):
        """
        Compute temporal coherence across video frames.

        Args:
            frame_features: List of frame feature vectors (each (512,))

        Returns:
            Dictionary with coherence metrics
        """
        n_frames = len(frame_features)
        feature_dim = len(frame_features[0])

        # Stack and flatten features
        stacked_features = np.stack(frame_features).flatten().astype(np.float32)

        # Compute temporal coherence
        coherence = self.bridge.compute_temporal_coherence(
            stacked_features, n_frames, feature_dim
        )

        # Reshape coherence to frame-level scores
        frame_coherence = np.mean(coherence.reshape(n_frames, feature_dim), axis=1)

        return {
            'frame_coherence': frame_coherence,
            'overall_coherence': np.mean(frame_coherence),
            'coherence_variance': np.var(frame_coherence)
        }

    def fuse_multimodal_features(self, text_features, visual_features, context=None):
        """
        Fuse text and visual features with optional context weighting.

        Args:
            text_features: Text embedding (512,)
            visual_features: Visual embedding (512,)
            context: Optional dict with 'modality' key for weighting

        Returns:
            Dictionary with fused features and weights
        """
        # Default equal weighting
        text_weight = 0.5
        visual_weight = 0.5

        # Adjust weights based on context if provided
        if context:
            if context.get('modality') == 'text':
                text_weight = 0.7
                visual_weight = 0.3
            elif context.get('modality') == 'visual':
                text_weight = 0.3
                visual_weight = 0.7

        # Fuse features
        fused = self.bridge.fuse_multimodal_features(
            text_features, visual_features, text_weight
        )

        return {
            'fused_features': fused,
            'text_weight': text_weight,
            'visual_weight': visual_weight,
            'context': context
        }

    def predict_next_state(self, action_vector, use_history=True):
        """
        Predict next world state given action vector.

        Args:
            action_vector: Action to apply (numpy array)
            use_history: Whether to use state history for prediction

        Returns:
            Predicted state vector
        """
        if not self.state_history:
            # No history, return zero state
            return np.zeros_like(action_vector)

        current_state = self.state_history[-1]

        # Predict next state
        predicted = self.bridge.predict_world_state(current_state, action_vector)

        # Update state history
        self.state_history.append(predicted)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

        return predicted

    def generate_dynamic_mesh(self, world_state, base_vertices, deformation_strength=0.2):
        """
        Generate dynamic mesh based on world state.

        Args:
            world_state: Current world state vector
            base_vertices: Base mesh vertices (N, 3)
            deformation_strength: Strength of deformation (0.0-1.0)

        Returns:
            Dynamic vertices (N, 3)
        """
        # Normalize deformation strength
        deformation_strength = np.clip(deformation_strength, 0.0, 1.0)

        # Scale world state influence
        scaled_state = world_state * deformation_strength

        # Generate dynamic mesh
        dynamic_vertices = self.bridge.generate_dynamic_mesh(scaled_state, base_vertices)

        return dynamic_vertices

    def enhance_galaxy_query(self, query_embedding, galaxy_embeddings, temperature=0.1):
        """
        Enhance galaxy query with temperature-scaled similarity.

        Args:
            query_embedding: Query vector (512,)
            galaxy_embeddings: Galaxy embeddings (N, 512)
            temperature: Temperature scaling factor

        Returns:
            Dictionary with resonance scores and top indices
        """
        resonance = self.bridge.enhance_galaxy_resonance(query_embedding, galaxy_embeddings)

        return {
            'resonance_scores': resonance,
            'top_indices': np.argsort(resonance)[::-1][:10],  # Top 10
            'temperature': temperature
        }

    def initialize_state(self, initial_features):
        """
        Initialize world model state with initial features.

        Args:
            initial_features: Initial state vector

        Returns:
            Initialized state
        """
        self.state_history = [initial_features.astype(np.float32)]
        return self.state_history[0]

    def get_state_context(self, window_size=5):
        """
        Get recent state context for prediction.

        Args:
            window_size: Number of recent states to consider

        Returns:
            Weighted average of recent states
        """
        if not self.state_history:
            return np.zeros(512)  # Default zero state

        # Get recent states
        recent_states = self.state_history[-window_size:]

        # Compute weighted average (more recent = higher weight)
        weights = np.linspace(0.5, 1.0, len(recent_states))
        weights = weights / np.sum(weights)

        context = np.average(recent_states, axis=0, weights=weights)

        return context

    def reset_state(self):
        """Reset world model state history."""
        self.state_history = []

    def get_stats(self):
        """Get world model statistics."""
        return {
            'state_history_length': len(self.state_history),
            'max_history': self.max_history,
            'current_state_norm': np.linalg.norm(self.state_history[-1]) if self.state_history else 0.0
        }
