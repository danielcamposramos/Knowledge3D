"""
Galaxy Injection Pipeline: PDF → Galaxy Nodes (Real-Time Learning)

Part of Week 1-2 implementation from Step7.1_FINAL.txt
Swarm collaboration: All AIs contribute
- Codex: Core implementation
- Grok: Edge case handling (malformed PDFs)
- Kimi: GPU optimization
- GLM: Embedding validation
- Qwen: Sleep integration hooks
- Claude: Tests + documentation

Key Features:
- Real-time PDF ingestion to Galaxy (seconds, not hours)
- RPN-powered embedding validation and similarity checks
- Chunk-based streaming (handles large PDFs)
- Honesty scoring for quality filtering
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json

import numpy as np
import cupy as cp

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False
    print("⚠️  PyPDF2 not available - install with: pip install PyPDF2")


class PDFGalaxyInjector:
    """
    Injects PDF content into Galaxy as semantic nodes.

    Uses RPN kernel for:
    - Embedding similarity checks (avoid duplicates)
    - Chunk quality scoring
    - Semantic coherence validation
    """

    def __init__(
        self,
        galaxy_path: str,
        chunk_size: int = 512,
        overlap: int = 128,
        min_quality: float = 0.5,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize PDF injector.

        Args:
            galaxy_path: Path to galaxy.glb file
            chunk_size: Characters per chunk
            overlap: Overlap between chunks (for context preservation)
            min_quality: Minimum quality score to inject
            similarity_threshold: Similarity threshold for deduplication
        """
        self.galaxy_path = Path(galaxy_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_quality = min_quality
        self.similarity_threshold = similarity_threshold

        # Load or initialize Galaxy
        from knowledge3d.spatial.galaxy import GalaxyGraph
        if self.galaxy_path.exists():
            self.galaxy = GalaxyGraph.load(str(self.galaxy_path))
        else:
            self.galaxy = GalaxyGraph.create_empty()

        # RPN executor for similarity and quality checks
        try:
            from knowledge3d.cranium.rpn_executor import get_rpn_executor
            from knowledge3d.cranium.clustering_rpn import compute_cosine_similarity_rpn
            from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_score_rpn
            self.rpn_executor = get_rpn_executor()
            self.compute_similarity = compute_cosine_similarity_rpn
            self.compute_quality = compute_honesty_score_rpn
            self._use_rpn = True
        except Exception as e:
            print(f"⚠️  RPN not available, using CPU fallback: {e}")
            self._use_rpn = False

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extract text from PDF, handling malformed PDFs (Grok's edge case handling).

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of text pages
        """
        if not _HAS_PYPDF2:
            raise RuntimeError("PyPDF2 required for PDF injection")

        pages = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                # Grok's edge case: Check if PDF is encrypted
                if reader.is_encrypted:
                    try:
                        reader.decrypt('')  # Try empty password
                    except:
                        raise ValueError(f"PDF is encrypted: {pdf_path}")

                # Extract all pages
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text and text.strip():
                            pages.append(text)
                    except Exception as e:
                        # Grok's edge case: Skip corrupted pages
                        print(f"⚠️  Skipping corrupted page {page_num}: {e}")
                        continue

        except Exception as e:
            # Grok's edge case: Handle malformed PDFs
            print(f"⚠️  Failed to read PDF {pdf_path}: {e}")
            raise

        return pages

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for context preservation.

        Args:
            text: Full text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)

            # Move forward with overlap
            start += (self.chunk_size - self.overlap)

        return chunks

    def compute_chunk_quality_rpn(
        self,
        chunk_text: str,
        embedding: np.ndarray
    ) -> float:
        """
        Compute quality score for text chunk using RPN.

        Quality components (RPN honesty scoring formula):
        - Correctness: Text coherence (via embedding norm)
        - Reasoning: Semantic richness (unique tokens ratio)
        - Uncertainty: Confidence (text length vs chunk_size)
        - Alignment: Domain relevance (embedding similarity to domain centroid)

        Args:
            chunk_text: Text content
            embedding: Embedding vector (normalized)

        Returns:
            Quality score [0, 1]
        """
        if not self._use_rpn:
            # CPU fallback: simple heuristic
            return min(1.0, len(chunk_text.split()) / 50.0)

        # Component 1: Correctness (embedding norm - should be close to 1 if normalized)
        emb_norm = float(np.linalg.norm(embedding))
        correctness = min(1.0, emb_norm)

        # Component 2: Reasoning (semantic richness - unique tokens ratio)
        tokens = chunk_text.lower().split()
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        reasoning = unique_ratio

        # Component 3: Uncertainty (confidence - longer chunks are more confident)
        confidence = min(1.0, len(chunk_text) / self.chunk_size)
        uncertainty = confidence

        # Component 4: Alignment (domain relevance - use embedding magnitude as proxy)
        alignment = min(1.0, np.abs(embedding).mean() * 10)  # Scale to [0,1]

        # RPN-powered quality score
        quality = self.compute_quality(
            correctness=correctness,
            reasoning=reasoning,
            uncertainty=uncertainty,
            alignment=alignment
        )

        return float(quality)

    def check_duplicate_rpn(
        self,
        embedding: np.ndarray,
        existing_embeddings: np.ndarray
    ) -> bool:
        """
        Check if embedding is duplicate using RPN cosine similarity.

        Args:
            embedding: New embedding to check
            existing_embeddings: Existing Galaxy embeddings (N, D)

        Returns:
            True if duplicate found (similarity > threshold)
        """
        if not self._use_rpn or len(existing_embeddings) == 0:
            return False

        # Use RPN to compute similarities with all existing
        from knowledge3d.cranium.clustering_rpn import compute_nearest_neighbors_rpn

        # Find nearest neighbor
        indices, similarities = compute_nearest_neighbors_rpn(
            query=embedding,
            embeddings=existing_embeddings,
            k=1
        )

        if len(similarities) > 0 and similarities[0] > self.similarity_threshold:
            return True

        return False

    def inject_pdf(
        self,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Inject PDF into Galaxy as semantic nodes.

        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata (author, title, etc.)

        Returns:
            Injection statistics
        """
        start_time = time.time()

        # Extract text from PDF
        print(f"📄 Extracting text from {pdf_path}...")
        pages = self.extract_text_from_pdf(pdf_path)
        print(f"   → Extracted {len(pages)} pages")

        # Chunk all pages
        print("✂️  Chunking text...")
        all_chunks = []
        for page_idx, page_text in enumerate(pages):
            chunks = self.chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'page': page_idx,
                    'chunk_idx': chunk_idx
                })
        print(f"   → Created {len(all_chunks)} chunks")

        # Get existing embeddings for deduplication
        existing_embeddings = self.galaxy.get_all_embeddings()
        if existing_embeddings is not None:
            existing_embeddings = np.array(existing_embeddings, dtype=np.float32)
        else:
            existing_embeddings = np.zeros((0, 768), dtype=np.float32)

        # Process chunks and inject
        print("🌌 Injecting to Galaxy...")
        injected_count = 0
        duplicate_count = 0
        low_quality_count = 0

        for chunk_data in all_chunks:
            chunk_text = chunk_data['text']

            # Generate embedding (using existing embedder)
            from knowledge3d.tools.embedders import embed_text_gpu
            embedding = embed_text_gpu(chunk_text)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize

            # Check for duplicates using RPN
            if self.check_duplicate_rpn(embedding, existing_embeddings):
                duplicate_count += 1
                continue

            # Compute quality using RPN
            quality = self.compute_chunk_quality_rpn(chunk_text, embedding)

            # Filter low-quality chunks
            if quality < self.min_quality:
                low_quality_count += 1
                continue

            # Create node ID
            node_id = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]

            # Inject to Galaxy
            node_metadata = {
                'source': str(pdf_path),
                'page': chunk_data['page'],
                'chunk_idx': chunk_data['chunk_idx'],
                'quality': float(quality),
                'type': 'pdf_chunk',
                **(metadata or {})
            }

            self.galaxy.add_node(
                node_id=node_id,
                embedding=embedding,
                content=chunk_text,
                metadata=node_metadata
            )

            injected_count += 1

            # Add to existing for future dedup checks
            existing_embeddings = np.vstack([existing_embeddings, embedding])

        # Save Galaxy
        self.galaxy.save(str(self.galaxy_path))

        elapsed = time.time() - start_time

        stats = {
            'pdf_path': str(pdf_path),
            'total_pages': len(pages),
            'total_chunks': len(all_chunks),
            'injected_nodes': injected_count,
            'duplicates_skipped': duplicate_count,
            'low_quality_skipped': low_quality_count,
            'elapsed_seconds': elapsed,
            'nodes_per_second': injected_count / max(0.001, elapsed),
            'galaxy_total_nodes': self.galaxy.node_count if hasattr(self.galaxy, 'node_count') else injected_count
        }

        print(f"\n✅ Injection complete!")
        print(f"   → {injected_count} nodes injected")
        print(f"   → {duplicate_count} duplicates skipped")
        print(f"   → {low_quality_count} low-quality skipped")
        print(f"   → {elapsed:.2f}s elapsed ({stats['nodes_per_second']:.1f} nodes/sec)")

        return stats


def inject_pdf_to_galaxy(
    pdf_path: str,
    galaxy_path: str = "viewer/public/galaxy/volatile_galaxy.glb",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to inject PDF to Galaxy.

    Args:
        pdf_path: Path to PDF file
        galaxy_path: Path to Galaxy GLB
        **kwargs: Additional arguments for PDFGalaxyInjector

    Returns:
        Injection statistics
    """
    injector = PDFGalaxyInjector(galaxy_path, **kwargs)
    return injector.inject_pdf(pdf_path)
