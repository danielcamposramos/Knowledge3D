import numpy as np
import logging

logger = logging.getLogger(__name__)

class CrossModalResonanceEngine:
    def __init__(self, fractal_emitter):
        self.fractal_emitter = fractal_emitter
        self.resonance_patterns = {
            ('text', 'image'): self.text_image_pattern,
            ('text', 'audio'): self.text_audio_pattern,
            ('image', 'audio'): self.image_audio_pattern,
            ('text', 'image', 'audio'): self.tri_modal_pattern
        }

    def apply_resonance_pattern(self, embeddings, modalities):
        modal_tuple = tuple(sorted(modalities))
        pattern_func = self.resonance_patterns.get(modal_tuple, self.default_pattern)
        
        try:
            return pattern_func(embeddings)
        except Exception as e:
            logger.error(f"Pattern application failed: {e}")
            return self.default_pattern(embeddings)

    def text_image_pattern(self, embeddings):
        text_emb = embeddings.get('text')
        image_emb = embeddings.get('image')
        if text_emb is not None and image_emb is not None:
            return self.fractal_emitter.create_cross_modal_links(text_emb, image_emb)
        return self.default_pattern(embeddings)

    def text_audio_pattern(self, embeddings):
        text_emb = embeddings.get('text')
        audio_emb = embeddings.get('audio')
        if text_emb is not None and audio_emb is not None:
            return (text_emb + audio_emb) / 2.0
        return self.default_pattern(embeddings)

    def image_audio_pattern(self, embeddings):
        image_emb = embeddings.get('image')
        audio_emb = embeddings.get('audio')
        if image_emb is not None and audio_emb is not None:
            return (image_emb + audio_emb) / 2.0
        return self.default_pattern(embeddings)

    def tri_modal_pattern(self, embeddings):
        emb_list = [embeddings.get(k) for k in ['text', 'image', 'audio']]
        valid_embs = [e for e in emb_list if e is not None]
        if valid_embs:
            return np.mean(valid_embs, axis=0)
        return self.default_pattern(embeddings)

    def default_pattern(self, embeddings):
        valid_embs = [e for e in embeddings.values() if e is not None]
        if valid_embs:
            return np.mean(valid_embs, axis=0)
        return np.array([])
