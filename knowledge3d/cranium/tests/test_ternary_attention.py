"""Tests for ternary attention masks."""

import numpy as np
import pytest

from knowledge3d.cranium.tools.ternary_attention import TernaryAttention

# Skip if GPU not available
try:
    from knowledge3d.cranium.bridges.sovereign_bridges import TernaryAttentionMask
    _probe = TernaryAttentionMask()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU/PTX not available")


def test_ternary_attention_basic():
    """Test basic ternary attention mask computation."""
    batch_size = 2
    seq_len = 16
    embed_dim = 128

    # Create random Q, K (normalized for stable similarities)
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    K = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)

    Q /= np.linalg.norm(Q, axis=2, keepdims=True)
    K /= np.linalg.norm(K, axis=2, keepdims=True)

    # Compute ternary masks
    attn = TernaryAttention(adaptive_thresholds=False, fixed_attract=0.5, fixed_repel=-0.2)
    masks = attn.compute_masks(Q, K)

    # Check output shape
    n_words = (seq_len * seq_len + 15) // 16
    assert masks.shape == (batch_size, n_words)
    assert masks.dtype == np.uint32

    # Unpack and check values
    unpacked = attn.unpack_masks(masks, seq_len)
    assert unpacked.shape == (batch_size, seq_len, seq_len)
    assert np.all((unpacked == -1) | (unpacked == 0) | (unpacked == 1))


def test_ternary_attention_adaptive():
    """Test adaptive threshold computation."""
    batch_size = 1
    seq_len = 32
    embed_dim = 64

    np.random.seed(123)
    Q = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    K = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)

    Q /= np.linalg.norm(Q, axis=2, keepdims=True)
    K /= np.linalg.norm(K, axis=2, keepdims=True)

    # Adaptive thresholds
    attn = TernaryAttention(
        adaptive_thresholds=True,
        attract_percentile=75.0,
        repel_percentile=25.0
    )

    masks = attn.compute_masks(Q, K)
    unpacked = attn.unpack_masks(masks, seq_len)

    # Should have roughly 25% attract, 25% repel, 50% neutral
    stats = attn.get_sparsity_stats(masks, seq_len)

    # Rough check (percentiles are approximate in kernel)
    assert 0.15 < stats["attract_fraction"] < 0.35  # ~25%
    assert 0.15 < stats["repel_fraction"] < 0.35    # ~25%
    assert 0.40 < stats["neutral_fraction"] < 0.60  # ~50%


def test_ternary_attention_sparsity():
    """Test sparsity statistics."""
    batch_size = 1
    seq_len = 8
    embed_dim = 32

    # Create highly similar Q, K (most should be +1)
    np.random.seed(456)
    base = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    Q = base + np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.1
    K = base + np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.1

    Q /= np.linalg.norm(Q, axis=2, keepdims=True)
    K /= np.linalg.norm(K, axis=2, keepdims=True)

    attn = TernaryAttention(adaptive_thresholds=False, fixed_attract=0.3, fixed_repel=-0.5)
    masks = attn.compute_masks(Q, K)

    stats = attn.get_sparsity_stats(masks, seq_len)

    # Most should be +1 (high similarity)
    assert stats["attract_fraction"] > 0.5
    assert stats["counts"][-1] + stats["counts"][0] + stats["counts"][1] == seq_len * seq_len


def test_ternary_attention_identity():
    """Test with identity Q=K (diagonal should be +1)."""
    batch_size = 1
    seq_len = 16
    embed_dim = 64

    np.random.seed(789)
    Q = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=2, keepdims=True)
    K = Q.copy()  # Identical

    attn = TernaryAttention(adaptive_thresholds=False, fixed_attract=0.9, fixed_repel=0.5)
    masks = attn.compute_masks(Q, K)
    unpacked = attn.unpack_masks(masks, seq_len)

    # Diagonal (self-attention) should be +1 (perfect similarity = 1.0)
    for i in range(seq_len):
        assert unpacked[0, i, i] == 1, f"Diagonal [{i},{i}] should be +1"


def test_ternary_attention_anti_identity():
    """Test with Q=-K (all should be -1)."""
    batch_size = 1
    seq_len = 8
    embed_dim = 32

    np.random.seed(101)
    Q = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=2, keepdims=True)
    K = -Q  # Opposite

    attn = TernaryAttention(adaptive_thresholds=False, fixed_attract=0.5, fixed_repel=-0.5)
    masks = attn.compute_masks(Q, K)
    unpacked = attn.unpack_masks(masks, seq_len)

    # All should be -1 (similarity = -1.0)
    assert np.all(unpacked == -1), "All should be -1 for anti-parallel vectors"


@pytest.mark.cuda
def test_ternary_attention_large_batch():
    """Test with larger batch (Tesla 18 instances)."""
    batch_size = 18  # Tesla 3-6-9 resonance
    seq_len = 32
    embed_dim = 128

    np.random.seed(202)
    Q = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    K = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)

    Q /= np.linalg.norm(Q, axis=2, keepdims=True)
    K /= np.linalg.norm(K, axis=2, keepdims=True)

    attn = TernaryAttention(adaptive_thresholds=True)
    masks = attn.compute_masks(Q, K)

    assert masks.shape[0] == batch_size

    # Check all batches have valid trits
    for b in range(batch_size):
        unpacked_b = attn.unpack_masks(masks[b:b+1], seq_len)
        assert np.all((unpacked_b == -1) | (unpacked_b == 0) | (unpacked_b == 1))
