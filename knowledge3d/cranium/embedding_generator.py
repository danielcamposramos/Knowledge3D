import argparse
import numpy as np


class DynamicEmbeddingGenerator:
    """Generate dynamic-dimension embeddings seeded deterministically by input.

    Media types map to suggested dimensions, but callers can override via
    the --dims CLI flag if needed.
    """

    def __init__(self):
        self.dim_suggestions = {
            "text": 128,
            "image": 512,
            "audio": 256,
            "video": 1024,
            "mixed": 2048,
        }

    def generate(self, input_text: str, media_type: str = "text", dims: int | None = None) -> np.ndarray:
        # Deterministic seed from input_text for reproducibility during prototyping
        np.random.seed(hash(input_text) % (2**32))
        dim = dims if dims is not None else self.dim_suggestions.get(media_type, 128)
        embedding = np.random.randn(dim).astype(np.float32)
        # Normalize for stability across downstream geometry operations
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm


def _cli():
    parser = argparse.ArgumentParser(description="Generate dynamic embeddings for Knowledge3D prototyping")
    parser.add_argument("--text", required=True, help="Input text to embed")
    parser.add_argument("--media", default="text", choices=["text", "image", "audio", "video", "mixed"], help="Media type")
    parser.add_argument("--dims", type=int, default=None, help="Override embedding dims")
    parser.add_argument("--out", default="embedding.npy", help="Output .npy file path")
    args = parser.parse_args()

    gen = DynamicEmbeddingGenerator()
    emb = gen.generate(args.text, args.media, args.dims)
    np.save(args.out, emb)
    print(f"Generated embedding dim={len(emb)} -> {args.out}")


if __name__ == "__main__":
    _cli()

