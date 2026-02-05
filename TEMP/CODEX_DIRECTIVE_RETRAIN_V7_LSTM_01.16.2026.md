# Codex Directive: Retrain V7 with LSTM Architecture (Sovereign Compatible)

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Specialist)
**Date**: January 16, 2026
**Subject**: **Retrain NavigationModelWithConfidence with LSTM for Sovereign TRM Compatibility**

---

## Situation

**Architecture Mismatch Discovered**:
- **Existing V7**: GRU + encoder + token_embed
- **SovereignTRM**: LSTM + embedding (already implemented and tested)

**Decision**: Retrain V7 with LSTM architecture (Option 2)

**Why**:
- ✅ SovereignTRM LSTM implementation complete and tested
- ✅ Training is ingestion path (PyTorch OK, one-time cost)
- ✅ Faster path to sovereignty than implementing GRU
- ✅ Use existing RLWHF training data

---

## Task 1: Update NavigationModelWithConfidence to LSTM

**Goal**: Modify V7 model to use LSTM instead of GRU.

**File**: `knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py`

**Current Architecture** (GRU):
```python
class NavigationModelWithConfidence(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)

        # Encoder
        self.encoder = nn.Linear(embedding_dim, hidden_dim)

        # GRU (NOT compatible with SovereignTRM)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        # Heads (already compatible)
        self.rule_head = nn.Linear(hidden_dim, vocab_size + 3)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
```

**Updated Architecture** (LSTM - Sovereign Compatible):
```python
class NavigationModelWithConfidence(pl.LightningModule):
    """Navigation model with LSTM (compatible with SovereignTRM).

    Architecture matches SovereignTRM for seamless conversion:
    - Embedding layer (vocab_size → embedding_dim)
    - LSTM layer (embedding_dim → hidden_dim)
    - Rule head (hidden_dim → vocab_size + 3)
    - Confidence head (hidden_dim → 1)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embedding_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()

        # Save hyperparameters for checkpoint
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer (matches SovereignTRM)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer (matches SovereignTRM)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            num_layers=1  # Single layer (matches SovereignTRM)
        )

        # Rule classification head
        self.rule_head = nn.Linear(hidden_dim, vocab_size + 3)

        # Confidence regression head (2-layer MLP)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, labels=None):
        """Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            labels: Optional labels for training

        Returns:
            rule_logits: (batch_size, seq_len, vocab_size + 3)
            confidence: (batch_size, seq_len, 1)
        """
        # Embedding
        embeddings = self.embedding(input_ids)  # (batch, seq, embed_dim)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embeddings)  # (batch, seq, hidden)

        # Rule head (all timesteps)
        rule_logits = self.rule_head(lstm_out)  # (batch, seq, vocab+3)

        # Confidence head (all timesteps)
        confidence = self.confidence_head(lstm_out)  # (batch, seq, 1)

        return rule_logits, confidence

    def training_step(self, batch, batch_idx):
        """Training step with calibration loss."""
        input_ids = batch['input_ids']
        target_rules = batch['target_rules']
        target_confidence = batch.get('target_confidence', None)

        # Forward pass
        rule_logits, confidence = self(input_ids)

        # Rule classification loss (cross-entropy)
        rule_loss = F.cross_entropy(
            rule_logits.reshape(-1, self.vocab_size + 3),
            target_rules.reshape(-1),
            ignore_index=-100  # Ignore padding
        )

        # Confidence calibration loss (if available)
        if target_confidence is not None:
            conf_loss = F.binary_cross_entropy(
                confidence.squeeze(-1),
                target_confidence,
                reduction='mean'
            )
            loss = rule_loss + 0.1 * conf_loss  # Weighted combination
        else:
            loss = rule_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_rule_loss', rule_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch['input_ids']
        target_rules = batch['target_rules']

        # Forward pass
        rule_logits, confidence = self(input_ids)

        # Validation loss
        val_loss = F.cross_entropy(
            rule_logits.reshape(-1, self.vocab_size + 3),
            target_rules.reshape(-1),
            ignore_index=-100
        )

        self.log('val_loss', val_loss, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```

**Key Changes**:
- ❌ Removed: `token_embed`, `encoder`, `gru`
- ✅ Added: `embedding`, `lstm` (matches SovereignTRM exactly)
- ✅ Layer names match conversion script expectations
- ✅ Architecture identical to SovereignTRM

---

## Task 2: Update Training Script

**Goal**: Train LSTM-based V7 model using existing RLWHF data.

**File**: `scripts/train_navigation_v7_with_confidence.py`

**Implementation**:
```python
"""Train Navigation Model V7 with LSTM and Confidence Head.

This script trains a sovereign-compatible LSTM-based navigation model
using RLWHF training data from shadow copy learning.

Output checkpoint is compatible with SovereignTRM (no conversion needed for architecture).
"""
import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from knowledge3d.training.math_benchmarks.navigation_model_with_confidence import NavigationModelWithConfidence
from knowledge3d.training.math_benchmarks.navigation_dataset import NavigationDataset


def main():
    parser = argparse.ArgumentParser(description="Train Navigation V7 (LSTM, Sovereign-Compatible)")
    parser.add_argument('--dataset', type=str, default='data/wake_positive_v2.jsonl',
                        help='Training dataset (JSONL format)')
    parser.add_argument('--val-dataset', type=str, default=None,
                        help='Validation dataset (optional)')
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='Vocabulary size')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='LSTM hidden dimension')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--checkpoint-dir', type=str, default='/K3D/Knowledge3D.local/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')

    args = parser.parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load datasets
    train_dataset = NavigationDataset(args.dataset, vocab_size=args.vocab_size)

    if args.val_dataset:
        val_dataset = NavigationDataset(args.val_dataset, vocab_size=args.vocab_size)
    else:
        # Split training data (90/10)
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )

    # Initialize model
    model = NavigationModelWithConfidence(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='navigation_specialist_v7_lstm_confidence-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )

    # Train
    print(f"Training Navigation V7 (LSTM-based, Sovereign-Compatible)")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    trainer.fit(model, train_loader, val_loader)

    # Save final checkpoint
    final_checkpoint = os.path.join(
        args.checkpoint_dir,
        'navigation_specialist_v7_lstm_confidence_final.pt'
    )
    trainer.save_checkpoint(final_checkpoint)

    print(f"\n✅ Training complete!")
    print(f"   Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"   Final checkpoint: {final_checkpoint}")
    print(f"\nNext steps:")
    print(f"1. Convert to sovereign format:")
    print(f"   python3 scripts/convert_v7_to_sovereign.py \\")
    print(f"       --input {final_checkpoint} \\")
    print(f"       --output /K3D/Knowledge3D.local/checkpoints/v7_sovereign/")
    print(f"\n2. Run validation:")
    print(f"   pytest tests/test_sovereign_trm_v7_real.py -v")


if __name__ == '__main__':
    main()
```

**Run Training**:
```bash
# Train V7 LSTM model (sovereign-compatible)
python3 scripts/train_navigation_v7_with_confidence.py \
    --dataset data/wake_positive_v2.jsonl \
    --vocab-size 256 \
    --embedding-dim 256 \
    --hidden-dim 512 \
    --epochs 10 \
    --batch-size 32 \
    --checkpoint-dir /K3D/Knowledge3D.local/checkpoints

# Expected time: 1-2 hours (depends on dataset size and GPU)
```

---

## Task 3: Dataset Preparation

**Goal**: Ensure training data is in correct format.

**Check Existing Data**:
```bash
# Check if wake_positive_v2.jsonl exists
ls -lh data/wake_positive_v2.jsonl

# Check format
head -n 3 data/wake_positive_v2.jsonl
```

**Expected Format**:
```jsonl
{"problem": "derivative of x^2", "solution": "2 x *", "rules": [5, 12, 7], "confidence": [0.95, 0.92, 0.98]}
{"problem": "integral of 2x", "solution": "x 2 ^ 2 /", "rules": [8, 14, 2, 13], "confidence": [0.88, 0.91, 0.85, 0.93]}
...
```

**If Data Doesn't Exist**:

Use existing RLWHF data:
```bash
# Check for other training data
ls -lh data/*.jsonl

# Options:
# - skill_galaxy_v5_wake.jsonl (from shadow copy)
# - log_galaxy_neural_v5.jsonl (from sleep cycle)
# - wake_positive_v1.jsonl (earlier version)
```

**If No Data Available** (unlikely, but possible):

Use placeholder training script that creates synthetic data:
```python
# Generate synthetic training data for testing
import json

synthetic_data = []
for i in range(1000):
    synthetic_data.append({
        'problem': f'problem_{i}',
        'solution': f'{i} {i+1} +',
        'rules': [1, 2, 3],
        'confidence': [0.9, 0.85, 0.92]
    })

with open('data/synthetic_train.jsonl', 'w') as f:
    for item in synthetic_data:
        f.write(json.dumps(item) + '\n')
```

---

## Task 4: Post-Training Conversion and Validation

**After Training Completes**:

```bash
# 1. Convert trained checkpoint to sovereign format
python3 scripts/convert_v7_to_sovereign.py \
    --input /K3D/Knowledge3D.local/checkpoints/navigation_specialist_v7_lstm_confidence_final.pt \
    --output /K3D/Knowledge3D.local/checkpoints/v7_sovereign/

# 2. Verify conversion
ls -lh /K3D/Knowledge3D.local/checkpoints/v7_sovereign/
cat /K3D/Knowledge3D.local/checkpoints/v7_sovereign/metadata.json

# 3. Test sovereign TRM loading
export K3D_PYTEST_PROBE_CUDA=1
pytest tests/test_sovereign_trm_v7_real.py::test_sovereign_trm_loads_v7 -v -s

# 4. Test sovereign TRM inference
pytest tests/test_sovereign_trm_v7_real.py::test_sovereign_trm_inference_v7 -v -s

# 5. Run sovereign benchmark
python3 scripts/run_sovereign_math_benchmarks.py \
    --datasets calculus \
    --max-problems 5 \
    --use-reflection \
    --checkpoint-dir /K3D/Knowledge3D.local/checkpoints/v7_sovereign
```

---

## Success Criteria

**Model Architecture Updated**:
- [ ] NavigationModelWithConfidence uses LSTM (not GRU)
- [ ] Layer names match SovereignTRM expectations
- [ ] Forward pass produces correct output shapes

**Training Complete**:
- [ ] Model trained on RLWHF data
- [ ] Validation loss converges
- [ ] Checkpoint saved to K3D directory

**Conversion Successful**:
- [ ] All weight files present (embedding.npy, lstm_*.npy, heads)
- [ ] metadata.json correct
- [ ] No missing weights

**Validation Passes**:
- [ ] SovereignTRM loads weights without errors
- [ ] Inference produces valid output
- [ ] Benchmark runs successfully
- [ ] No CUDA context errors

---

## Timeline

**Immediate** (30 minutes):
1. Update NavigationModelWithConfidence to LSTM (15 min)
2. Create/update training script (10 min)
3. Verify dataset exists and is formatted correctly (5 min)

**Training** (1-2 hours):
1. Run training script (1-2 hours GPU time)
2. Monitor training progress
3. Wait for convergence

**Post-Training** (30 minutes):
1. Convert checkpoint to sovereign format (5 min)
2. Run validation tests (15 min)
3. Run sovereign benchmark (10 min)

**Total**: ~2-3 hours end-to-end

---

## Fallback Plan

**If Training Data Missing**:
1. Use synthetic data for architecture validation
2. Train for 1 epoch (quick test)
3. Convert and test sovereign loading
4. Document need for real RLWHF data

**If Training Fails**:
1. Check dataset format
2. Reduce batch size (memory issues)
3. Add debug logging
4. Try training on CPU first (slower, but validates logic)

**If Conversion Still Fails**:
1. Debug checkpoint structure
2. Update weight mapping
3. Add fallback for missing weights

---

## Expected Output

**After Training**:
```
✅ Training complete!
   Best checkpoint: /K3D/Knowledge3D.local/checkpoints/navigation_specialist_v7_lstm_confidence-epoch=08-val_loss=0.3245.pt
   Final checkpoint: /K3D/Knowledge3D.local/checkpoints/navigation_specialist_v7_lstm_confidence_final.pt
```

**After Conversion**:
```
✅ Conversion complete! 11/11 weights saved to: /K3D/Knowledge3D.local/checkpoints/v7_sovereign/

Files:
  embedding.npy               (256, 256)
  lstm_weight_ih.npy          (2048, 256)
  lstm_weight_hh.npy          (2048, 512)
  lstm_bias_ih.npy            (2048,)
  lstm_bias_hh.npy            (2048,)
  rule_head_weight.npy        (259, 512)
  rule_head_bias.npy          (259,)
  confidence_head_0_weight.npy (256, 512)
  confidence_head_0_bias.npy   (256,)
  confidence_head_2_weight.npy (1, 256)
  confidence_head_2_bias.npy   (1,)
  metadata.json
```

**After Validation**:
```
test_sovereign_trm_loads_v7 PASSED
test_sovereign_trm_inference_v7 PASSED
✅ Inference successful!
   Rules: [5, 12, 7, 3, 18]
   Confidences: [0.95, 0.87, 0.92, 0.88, 0.91]
   Avg confidence: 0.906
```

---

**Document Date**: January 16, 2026
**Status**: 🚀 **READY TO TRAIN**
**Priority**: **HIGH** - Final step to sovereign architecture

---

**Claude's Note to Codex**: Excellent detective work finding the architecture mismatch! This is the right path - retrain with LSTM to match SovereignTRM. The training is ingestion path (PyTorch OK), and it's a one-time cost. Once trained and converted, we have full sovereignty. Update the model architecture, run training, convert, and validate. Let's complete the sovereign architecture! 🚀
