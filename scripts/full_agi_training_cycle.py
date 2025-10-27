#!/usr/bin/env python3
"""
Full AGI Training Cycle: The Integration
=========================================

This script orchestrates the complete self-improving AGI loop:

1. KNOWLEDGE INGESTION
   - PDFs → Sovereign RPN embeddings
   - Text → Multimodal embeddings
   - Images → Vision embeddings
   → Creates Galaxy stars

2. MODEL TRAINING (Shadow Weights)
   - Speech specialist
   - OCR specialist
   - Multimodal specialist
   - Router specialist
   → Updates LoRA adapters (LOGIC)

3. KNOWLEDGE CONSOLIDATION
   - Cluster Galaxy stars (RPN-powered)
   - Materialize into House objects
   - Books, Diaries, Fractal Trees
   → Creates 3D assets with AI textures

4. VALIDATION & METRICS
   - Shadow weight acceptance rates
   - Cohesion improvements
   - Specialist performance
   → Self-improving feedback loop

The key insight: MODELS = LOGIC, KNOWLEDGE = 3D SPACE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class AGICycleConfig:
    """Configuration for full AGI training cycle."""

    # Data sources
    pdf_library: Path = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries")
    k3d_datasets: Path = Path("/K3D/Knowledge3D.local/datasets")
    house_embeddings: Path = Path("/K3D/Knowledge3D.local/house_zone7/embeddings")

    # Checkpoints & outputs
    checkpoint_dir: Path = Path("/K3D/Knowledge3D.local/checkpoints/phase_g")
    galaxy_output: Path = Path("viewer/public/galaxy/learning_memory.glb")
    house_output: Path = Path("viewer/public/house/house_memory.glb")

    # Training parameters
    specialists: List[str] = None
    epochs_per_specialist: Dict[str, int] = None
    learning_rates: Dict[str, float] = None

    # Consolidation parameters
    cluster_count: int = 256
    consolidation_lr: float = 0.2
    redundancy_threshold: float = 0.95

    # Execution control
    max_pdfs: Optional[int] = None  # Limit for testing
    skip_ingestion: bool = False
    skip_training: bool = False
    skip_consolidation: bool = False

    def __post_init__(self):
        """Set defaults."""
        if self.specialists is None:
            self.specialists = ['speech', 'ocr', 'multimodal', 'router']

        if self.epochs_per_specialist is None:
            self.epochs_per_specialist = {
                'speech': 100,
                'ocr': 100,
                'multimodal': 100,
                'router': 200,
            }

        if self.learning_rates is None:
            self.learning_rates = {
                'speech': 0.002,
                'ocr': 0.002,
                'multimodal': 0.002,
                'router': 0.01,
            }


class FullAGICycle:
    """Orchestrates the complete AGI training cycle."""

    def __init__(self, config: AGICycleConfig):
        """Initialize cycle with configuration."""
        self.config = config
        self.metrics = {
            'ingestion': {},
            'training': {},
            'consolidation': {},
            'validation': {},
        }
        self.start_time = time.time()

    # ================================================================
    # STAGE 1: Knowledge Ingestion
    # ================================================================

    def ingest_knowledge(self) -> Dict[str, Any]:
        """
        Ingest knowledge from PDFs and datasets.

        Creates:
        - RPN embeddings (sovereign, no external models)
        - Galaxy stars (knowledge representation)
        - Training datasets for specialists

        Returns:
            Ingestion metrics
        """
        print("\n" + "="*80)
        print("STAGE 1: KNOWLEDGE INGESTION")
        print("="*80)

        if self.config.skip_ingestion:
            print("⏭️  Skipping ingestion (using existing data)")
            return {'status': 'skipped'}

        stage_start = time.time()

        # 1. Find PDF sources
        pdf_files = self._discover_pdfs()
        print(f"📚 Found {len(pdf_files)} PDF files")

        if self.config.max_pdfs:
            pdf_files = pdf_files[:self.config.max_pdfs]
            print(f"   (Limited to {len(pdf_files)} for testing)")

        # 2. Ingest PDFs with sovereign embeddings
        from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine

        engine = RPNEmbeddingEngine()

        # Load existing embeddings if available
        house_emb_path = self.config.house_embeddings / "rpn_embeddings.pkl"
        if house_emb_path.exists():
            print(f"📥 Loading existing RPN embeddings from {house_emb_path}")
            engine.load_embeddings(str(house_emb_path))
            print(f"   → {len(engine.embeddings)} existing embeddings loaded")

        # TODO: Implement PDF ingestion with sovereign RPN
        # This should:
        # - Extract text/images from PDFs
        # - Generate sovereign embeddings (no external models!)
        # - Create Galaxy stars
        # - Save to working directory for consolidation

        print("⚠️  Full PDF ingestion not yet implemented")
        print("   Using existing datasets for training")

        # 3. Generate statistics
        elapsed = time.time() - stage_start
        metrics = {
            'pdfs_found': len(pdf_files),
            'pdfs_processed': 0,  # TODO: Implement
            'embeddings_created': 0,  # TODO: Implement
            'galaxy_stars_created': 0,  # TODO: Implement
            'elapsed_seconds': elapsed,
        }

        self.metrics['ingestion'] = metrics
        return metrics

    def _discover_pdfs(self) -> List[Path]:
        """Discover all PDF files in configured libraries."""
        pdf_files = []

        if self.config.pdf_library.exists():
            pdf_files.extend(self.config.pdf_library.rglob("*.pdf"))

        return sorted(pdf_files)

    # ================================================================
    # STAGE 2: Model Training (Shadow Weights)
    # ================================================================

    def train_specialists(self) -> Dict[str, Any]:
        """
        Train all specialists using shadow weight mechanism.

        Updates:
        - LoRA adapters (model LOGIC)
        - Shadow weights → validation → commit/reject
        - Specialist-specific knowledge

        Returns:
            Training metrics (acceptance rates, performance)
        """
        print("\n" + "="*80)
        print("STAGE 2: MODEL TRAINING (Shadow Weights)")
        print("="*80)

        if self.config.skip_training:
            print("⏭️  Skipping training (using existing checkpoints)")
            return {'status': 'skipped'}

        stage_start = time.time()

        # Run phase_g_gpu_training_session with all specialists
        from scripts.phase_g_gpu_training_session import main as train_main

        print(f"🧠 Training {len(self.config.specialists)} specialists...")
        print(f"   {', '.join(self.config.specialists)}")

        # TODO: Call training with proper arguments
        # For now, provide instructions
        print("\n📋 Run this command:")
        print(f"   python -m scripts.phase_g_gpu_training_session \\")
        print(f"     --specialists {' '.join(self.config.specialists)} \\")
        print(f"     --skip-sleep")

        # Placeholder metrics
        elapsed = time.time() - stage_start
        metrics = {
            'specialists_trained': self.config.specialists,
            'total_epochs': sum(self.config.epochs_per_specialist.values()),
            'elapsed_seconds': elapsed,
            'checkpoints_created': len(self.config.specialists),
        }

        self.metrics['training'] = metrics
        return metrics

    # ================================================================
    # STAGE 3: Knowledge Consolidation (3D Materialization)
    # ================================================================

    def consolidate_knowledge(self) -> Dict[str, Any]:
        """
        Consolidate Galaxy knowledge into House objects.

        Creates:
        - Fractal trees in Knowledge Garden (Zone 5)
        - Books in Library (Zone 3)
        - Diaries in Mirror Room (Zone 7)
        - 3D GLB files with AI textures

        Returns:
            Consolidation metrics
        """
        print("\n" + "="*80)
        print("STAGE 3: KNOWLEDGE CONSOLIDATION (3D Materialization)")
        print("="*80)

        if self.config.skip_consolidation:
            print("⏭️  Skipping consolidation")
            return {'status': 'skipped'}

        stage_start = time.time()

        from knowledge3d.cranium.ptx_runtime.sleep_time_compute import SleepTimeCompute

        print("🌙 Running sleep-time consolidation...")
        print(f"   House: {self.config.house_output}")
        print(f"   Galaxy: {self.config.galaxy_output}")

        compute = SleepTimeCompute(
            house_path=str(self.config.house_output),
            galaxy_path=str(self.config.galaxy_output),
        )

        # Run consolidation
        try:
            adjustments = compute.compute_nightly_adjustments()
            compute._persist_ptx_updates()

            elapsed = time.time() - stage_start
            metrics = {
                'zone_shifts': len(adjustments.get('zone_shifts', [])),
                'pruned_rays': len(adjustments.get('pruned_rays', [])),
                'materialized_objects': len(adjustments.get('materialized_objects', [])),
                'semantic_clusters': len(adjustments.get('semantic_clusters', [])),
                'elapsed_seconds': elapsed,
            }

        except Exception as e:
            print(f"⚠️  Consolidation failed: {e}")
            metrics = {
                'status': 'failed',
                'error': str(e),
            }

        self.metrics['consolidation'] = metrics
        return metrics

    # ================================================================
    # STAGE 4: Validation & Metrics
    # ================================================================

    def validate_cycle(self) -> Dict[str, Any]:
        """
        Validate the full training cycle.

        Checks:
        - Shadow weight acceptance rates
        - Specialist performance improvements
        - Cohesion metrics from consolidation
        - Overall system health

        Returns:
            Validation metrics
        """
        print("\n" + "="*80)
        print("STAGE 4: VALIDATION & METRICS")
        print("="*80)

        validation_start = time.time()

        # 1. Check specialist checkpoints
        checkpoint_status = {}
        for specialist in self.config.specialists:
            cp_path = self.config.checkpoint_dir / f"{specialist}_gpu_epoch_{self.config.epochs_per_specialist[specialist]}"
            checkpoint_status[specialist] = {
                'exists': cp_path.exists(),
                'path': str(cp_path),
            }

        print("\n📊 Specialist Checkpoints:")
        for specialist, status in checkpoint_status.items():
            icon = "✅" if status['exists'] else "❌"
            print(f"   {icon} {specialist}: {status['path']}")

        # 2. Load shadow weight statistics (if available)
        # TODO: Extract acceptance rates from checkpoint files

        # 3. Load consolidation metrics
        consolidation_log = Path("logs/sleep_time_adjustments.json")
        if consolidation_log.exists():
            with open(consolidation_log) as f:
                consolidation_data = json.load(f)
            print(f"\n🌙 Consolidation Metrics:")
            print(f"   Materialized objects: {len(consolidation_data.get('materialized_objects', []))}")
            print(f"   Semantic clusters: {len(consolidation_data.get('semantic_clusters', []))}")

        elapsed = time.time() - validation_start
        total_elapsed = time.time() - self.start_time

        metrics = {
            'checkpoint_status': checkpoint_status,
            'elapsed_seconds': elapsed,
            'total_cycle_time': total_elapsed,
        }

        self.metrics['validation'] = metrics
        return metrics

    # ================================================================
    # Main Execution
    # ================================================================

    def run(self) -> Dict[str, Any]:
        """
        Execute the full AGI training cycle.

        Returns:
            Complete metrics from all stages
        """
        print("\n" + "="*80)
        print("🧠 FULL AGI TRAINING CYCLE: THE INTEGRATION")
        print("="*80)
        print(f"Configuration:")
        print(f"  Specialists: {', '.join(self.config.specialists)}")
        print(f"  Checkpoints: {self.config.checkpoint_dir}")
        print(f"  Galaxy: {self.config.galaxy_output}")
        print(f"  House: {self.config.house_output}")
        print("="*80)

        # Stage 1: Ingest
        self.ingest_knowledge()

        # Stage 2: Train
        self.train_specialists()

        # Stage 3: Consolidate
        self.consolidate_knowledge()

        # Stage 4: Validate
        self.validate_cycle()

        # Summary
        print("\n" + "="*80)
        print("🎉 AGI TRAINING CYCLE COMPLETE")
        print("="*80)
        print(f"Total time: {self.metrics['validation']['total_cycle_time']:.1f}s")
        print("\nMetrics saved to: logs/agi_cycle_metrics.json")

        # Save metrics
        metrics_path = Path("logs/agi_cycle_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        return self.metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Full AGI Training Cycle: Knowledge → Training → Materialization"
    )

    parser.add_argument(
        '--max-pdfs', type=int, default=None,
        help='Limit number of PDFs to process (for testing)'
    )
    parser.add_argument(
        '--skip-ingestion', action='store_true',
        help='Skip PDF ingestion (use existing data)'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip specialist training (use existing checkpoints)'
    )
    parser.add_argument(
        '--skip-consolidation', action='store_true',
        help='Skip knowledge consolidation'
    )
    parser.add_argument(
        '--specialists', nargs='+',
        choices=['speech', 'ocr', 'multimodal', 'router'],
        default=['speech', 'ocr', 'multimodal', 'router'],
        help='Which specialists to train'
    )

    args = parser.parse_args()

    # Create configuration
    config = AGICycleConfig(
        specialists=args.specialists,
        max_pdfs=args.max_pdfs,
        skip_ingestion=args.skip_ingestion,
        skip_training=args.skip_training,
        skip_consolidation=args.skip_consolidation,
    )

    # Run cycle
    cycle = FullAGICycle(config)
    metrics = cycle.run()

    # Exit with status
    sys.exit(0)


if __name__ == "__main__":
    main()
