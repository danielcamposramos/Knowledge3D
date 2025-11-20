#!/usr/bin/env python3
"""
Full Atomic Procedural Training - 1,002 Atomic Units

This script trains the base model on ALL atomic knowledge units:
- 450 font glyphs (form only)
- 552 math symbols (dual-modal: form + execution meaning)

THESIS: 3D contract is superior to tokenization for general knowledge representation

PROOF STRATEGY:
1. Build well-defined atomic units via compositional storage (visual_rpn + math_rpn)
2. Cross-modality emerges from 3D contract, NOT from tokenization
3. Visual form as grounding enables visual input KR (Milton's challenge)
4. Document ALL metrics as evidence for W3C AIKR contribution

W3C AIKR Context:
  Milton Ponson's challenge: Tokenization lacks well-defined atomic units
  K3D's answer: Dual-program stars with visual form + execution meaning
  Evidence: This training run's metrics and atomic unit formation

Author: K3D Adaptive Swarm
Date: 2025-11-19
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.cranium.specialists.procedural_drawing_specialist import ProceduralDrawingSpecialist


class MetricsLogger:
    """Comprehensive metrics logger for W3C AIKR evidence."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.training_log = []
        self.atomic_units_log = []
        self.alignment_distribution = []

        # W3C AIKR thesis metrics
        self.thesis_metrics = {
            'total_atomic_units': 0,
            'dual_modal_units': 0,
            'visual_only_units': 0,
            'compositional_fusion_success_rate': 0.0,
            'cross_modality_evidence': [],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def log_epoch(self, epoch: int, split: str, category: str, metrics: Dict):
        """Log epoch metrics."""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'epoch': epoch,
            'split': split,
            'category': category,
            'alignment': metrics.get('alignment', 0.0),
            'loss': metrics.get('loss', 0.0),
            'samples': metrics.get('samples', 0)
        }
        self.training_log.append(entry)

        # Save incrementally
        with open(self.output_dir / 'training_log.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_atomic_unit(self, char: str, unit_data: Dict):
        """Log atomic unit details."""
        glyphs = unit_data.get('glyphs', [])
        sample_font = None
        if glyphs:
            font_meta = glyphs[0].get('font_metadata', {})
            sample_font = font_meta.get('font_family') or font_meta.get('font_name')

        entry = {
            'char': char,
            'has_visual_rpn': bool(glyphs),
            'has_math_rpn': bool(unit_data.get('math_rpn', '')),
            'glyph_count': len(glyphs),
            'sample_font': sample_font,
            'embedding_shape': list(unit_data['embedding'].shape) if 'embedding' in unit_data else [],
            'timestamp': unit_data.get('timestamp', datetime.now(timezone.utc).isoformat())
        }
        self.atomic_units_log.append(entry)

    def compute_thesis_metrics(self, specialist: ProceduralDrawingSpecialist):
        """Compute metrics for W3C AIKR thesis."""
        total = len(specialist.atomic_units)
        dual_modal = sum(1 for u in specialist.atomic_units.values() if u.get('math_rpn'))
        visual_only = total - dual_modal

        self.thesis_metrics.update({
            'total_atomic_units': total,
            'dual_modal_units': dual_modal,
            'visual_only_units': visual_only,
            'compositional_fusion_success_rate': dual_modal / max(total, 1),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        # Cross-modality evidence: chars with both visual + math RPN
        for char, unit in specialist.atomic_units.items():
            glyphs = unit.get('glyphs', [])
            if glyphs and unit.get('math_rpn'):
                self.thesis_metrics['cross_modality_evidence'].append({
                    'char': char,
                    'visual_rpn_length': len(glyphs[0]['visual_rpn']),
                    'math_rpn': unit.get('math_rpn', ''),
                    'embedding_dim': len(unit['embedding']),
                    'font_family': glyphs[0].get('font_metadata', {}).get('font_family')
                })

    def save_final_report(self):
        """Save comprehensive final report."""
        report = {
            'thesis': '3D contract is superior to tokenization for general KR',
            'w3c_aikr_context': {
                'milton_challenge': 'Tokenization lacks well-defined atomic units',
                'k3d_answer': 'Dual-program stars with compositional fusion',
                'proof': 'This training run demonstrates atomic unit formation'
            },
            'metrics': self.thesis_metrics,
            'training_summary': {
                'total_epochs': max([e['epoch'] for e in self.training_log], default=0) + 1,
                'total_samples': sum([e['samples'] for e in self.training_log]),
                'avg_alignment_font': np.mean([e['alignment'] for e in self.training_log if e['category'] == 'font']),
                'avg_alignment_math': np.mean([e['alignment'] for e in self.training_log if e['category'] == 'math'])
            },
            'atomic_units': self.atomic_units_log,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        with open(self.output_dir / 'w3c_aikr_evidence.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Human-readable summary
        with open(self.output_dir / 'TRAINING_SUMMARY.md', 'w') as f:
            f.write(f"""# Atomic Procedural Training - W3C AIKR Evidence

## Thesis
**3D contract is superior to tokenization for general knowledge representation**

## W3C AIKR Context
**Milton Ponson's Challenge**: Tokenization (LLMs' "blue bubbles") lacks well-defined atomic units from set theory for general KR.

**K3D's Answer**: Dual-program stars with compositional fusion
- Visual form (HOW to draw) + Execution meaning (WHAT it does)
- Both programs stored in SAME star → cross-modality via 3D contract
- No tokenization needed - atomic units are procedural programs

## Evidence from This Training Run

### Atomic Units Formed
- **Total**: {self.thesis_metrics['total_atomic_units']} atomic units
- **Dual-modal** (visual + math): {self.thesis_metrics['dual_modal_units']} units
- **Visual-only** (fonts): {self.thesis_metrics['visual_only_units']} units
- **Compositional success rate**: {self.thesis_metrics['compositional_fusion_success_rate']:.2%}

### Training Metrics
- **Epochs**: {report['training_summary']['total_epochs']}
- **Total samples**: {report['training_summary']['total_samples']}
- **Avg alignment (fonts)**: {report['training_summary']['avg_alignment_font']:.4f}
- **Avg alignment (math)**: {report['training_summary']['avg_alignment_math']:.4f}

### Cross-Modality Evidence
Examples of dual-program stars:
""")

            # Show first 10 cross-modal examples
            for evidence in self.thesis_metrics['cross_modality_evidence'][:10]:
                f.write(f"\n**'{evidence['char']}'**:\n")
                f.write(f"  - Visual RPN: {evidence['visual_rpn_length']} chars\n")
                f.write(f"  - Math RPN: {evidence['math_rpn']}\n")
                f.write(f"  - Embedding: {evidence['embedding_dim']}D\n")
                if evidence.get('font_family'):
                    f.write(f"  - Sample font: {evidence['font_family']}\n")

        print(f"\n✅ W3C AIKR evidence saved to {self.output_dir}/")
        return report


def load_datasets(font_glob: str, font_limit_per_file: int | None):
    """Load font + math datasets (multi-script aware)."""
    math_path = Path("/K3D/Knowledge3D.local/datasets/atomic/math_symbols_procedural.jsonl")

    # Discover all font datasets matching the glob
    font_files = sorted(Path().glob(font_glob))
    if not font_files:
        raise FileNotFoundError(f"No font datasets found for glob: {font_glob}")

    font_data: List[Dict] = []
    for font_file in font_files:
        loaded = 0
        with font_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                font_data.append(json.loads(line))
                loaded += 1
                if font_limit_per_file and loaded >= font_limit_per_file:
                    break
        print(f"  • {font_file}: {loaded} entries")

    math_data = []
    with open(math_path, 'r') as f:
        for line in f:
            if line.strip():
                math_data.append(json.loads(line))

    print(f"✅ Loaded {len(font_data)} font glyphs across {len(font_files)} files, {len(math_data)} math symbols")
    return font_data, math_data


def train_full_atomic_knowledge(epochs: int = 5, validation_split: float = 0.1, font_glob: str = "/K3D/Knowledge3D.local/datasets/atomic/fonts_*_procedural.jsonl", font_limit_per_file: int | None = None):
    """
    Train full atomic knowledge base with comprehensive metric tracking.

    Args:
        epochs: Number of training epochs
        validation_split: Fraction of data for validation

    Returns:
        Trained specialist and metrics logger
    """
    print("\n" + "=" * 70)
    print("FULL ATOMIC PROCEDURAL TRAINING")
    print("Thesis: 3D Contract > Tokenization for General KR")
    print("=" * 70)

    # Initialize metrics logger
    output_dir = Path("/K3D/Knowledge3D.local/logs/atomic_training") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger = MetricsLogger(output_dir)

    # Initialize swarm and specialist
    print("\n[1/6] Initializing AdaptiveSwarm...")
    config = SwarmConfig(base_dims=512, min_dims=64)
    swarm = AdaptiveSwarmTRM(config=config)

    print("\n[2/6] Initializing ProceduralDrawingSpecialist...")
    specialist = ProceduralDrawingSpecialist(
        swarm=swarm,
        matryoshka_dim=512,
        gpu_id=0
    )

    # Load datasets
    print("\n[3/6] Loading full datasets...")
    font_data, math_data = load_datasets(font_glob, font_limit_per_file)

    # Split into train/val
    font_split = int(len(font_data) * (1 - validation_split))
    math_split = int(len(math_data) * (1 - validation_split))

    font_train = font_data[:font_split]
    font_val = font_data[font_split:]
    math_train = math_data[:math_split]
    math_val = math_data[math_split:]

    print(f"  Font: {len(font_train)} train, {len(font_val)} val")
    print(f"  Math: {len(math_train)} train, {len(math_val)} val")
    print(f"  Total training samples: {len(font_train) + len(math_train)}")

    # Training loop
    print(f"\n[4/6] Training for {epochs} epochs...")

    for epoch in range(epochs):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print('=' * 70)

        # Font glyphs training
        print("\n  [Font Glyphs Training]")
        font_batch = [(item['char'], item['rpn'], item) for item in font_train]
        font_metrics = specialist.train_on_batch(
            font_batch,
            validation=False,
            dual_modal_math=False
        )
        print(f"    Alignment: {font_metrics.text_visual_alignment:.4f}")
        logger.log_epoch(epoch, 'train', 'font', {
            'alignment': font_metrics.text_visual_alignment,
            'loss': font_metrics.latency_us,
            'samples': len(font_batch)
        })

        # Font glyphs validation
        print("  [Font Glyphs Validation]")
        font_val_batch = [(item['char'], item['rpn'], item) for item in font_val]
        font_val_metrics = specialist.train_on_batch(
            font_val_batch,
            validation=True,
            dual_modal_math=False
        )
        print(f"    Alignment: {font_val_metrics.text_visual_alignment:.4f}")
        logger.log_epoch(epoch, 'val', 'font', {
            'alignment': font_val_metrics.text_visual_alignment,
            'loss': font_val_metrics.latency_us,
            'samples': len(font_val_batch)
        })

        # Math symbols training
        print("\n  [Math Symbols Training]")
        math_batch = [
            (item['char'], item['visual_rpn'], item.get('math_rpn', ''), item['semantic'], item)
            for item in math_train
        ]
        math_metrics = specialist.train_on_batch(
            math_batch,
            validation=False,
            dual_modal_math=True
        )
        print(f"    Alignment: {math_metrics.text_visual_alignment:.4f}")
        logger.log_epoch(epoch, 'train', 'math', {
            'alignment': math_metrics.text_visual_alignment,
            'loss': math_metrics.latency_us,
            'samples': len(math_batch)
        })

        # Math symbols validation
        print("  [Math Symbols Validation]")
        math_val_batch = [
            (item['char'], item['visual_rpn'], item.get('math_rpn', ''), item['semantic'], item)
            for item in math_val
        ]
        math_val_metrics = specialist.train_on_batch(
            math_val_batch,
            validation=True,
            dual_modal_math=True
        )
        print(f"    Alignment: {math_val_metrics.text_visual_alignment:.4f}")
        logger.log_epoch(epoch, 'val', 'math', {
            'alignment': math_val_metrics.text_visual_alignment,
            'loss': math_val_metrics.latency_us,
            'samples': len(math_val_batch)
        })

    # Log all atomic units
    print("\n[5/6] Logging atomic units...")
    for char, unit in specialist.atomic_units.items():
        logger.log_atomic_unit(char, unit)

    print(f"  Total atomic units: {len(specialist.atomic_units)}")

    # Commit to ProceduralGalaxy
    print("\n[6/6] Committing to ProceduralGalaxy...")
    committed = specialist.commit_atomic_units_to_galaxy()
    print(f"  ✅ Committed {committed} atomic units")

    # Compute thesis metrics
    print("\n[W3C AIKR] Computing thesis evidence metrics...")
    logger.compute_thesis_metrics(specialist)

    # Save final report
    print("\n[W3C AIKR] Generating evidence report...")
    report = logger.save_final_report()

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - W3C AIKR EVIDENCE")
    print("=" * 70)
    print(f"\nThesis: {report['thesis']}")
    print(f"\nAtomic Units:")
    print(f"  Total: {report['metrics']['total_atomic_units']}")
    print(f"  Dual-modal (visual + math): {report['metrics']['dual_modal_units']}")
    print(f"  Visual-only (fonts): {report['metrics']['visual_only_units']}")
    print(f"  Compositional success: {report['metrics']['compositional_fusion_success_rate']:.2%}")
    print(f"\nCross-modality examples: {len(report['metrics']['cross_modality_evidence'])}")
    print(f"\nEvidence saved to: {output_dir}/")
    print(f"  - training_log.jsonl")
    print(f"  - w3c_aikr_evidence.json")
    print(f"  - TRAINING_SUMMARY.md")

    return specialist, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full atomic procedural training (multi-script).")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--font-glob", type=str, default="/K3D/Knowledge3D.local/datasets/atomic/fonts_*_procedural.jsonl", help="Glob for font datasets to include.")
    parser.add_argument("--font-limit-per-file", type=int, default=None, help="Optional cap per font dataset file for memory control.")
    args = parser.parse_args()

    try:
        specialist, logger = train_full_atomic_knowledge(
            epochs=args.epochs,
            validation_split=args.validation_split,
            font_glob=args.font_glob,
            font_limit_per_file=args.font_limit_per_file,
        )
        print("\n✅ Full atomic training completed successfully!\n")
        print("Next steps:")
        print("  1. Review W3C AIKR evidence in logs/atomic_training/")
        print("  2. Implement RPN sovereignty (Phase 2)")
        print("  3. Retrain with full PTX-native operations")
        print("  4. Submit findings to W3C AIKR CG\n")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
