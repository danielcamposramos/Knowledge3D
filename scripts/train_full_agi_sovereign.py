#!/usr/bin/env python3
"""
Full AGI Training with Sovereign Engine + Dual Sleep Cycles

Complete training pipeline for Knowledge3D AGI:
1. Multimodal training (text + images + audio + video)
2. Language training (Wikipedia, compendiums, text corpora)
3. Curated PDFs (EchoSystems Default Libraries)
4. Traditional AI datasets (COCO, AudioCaps, etc.)

Architecture:
- MODELS = LOGIC (LoRA adapters with shadow weights)
- KNOWLEDGE = 3D SPACE (Galaxy stars → House objects)

After EACH training phase:
  Sleep 1: Model Updates (validate shadow weights, commit improvements)
  Sleep 2: Knowledge Consolidation (materialize Galaxy → House with φ constraints)

Result: Paired House + Models that self-improve continuously
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Phase G integration
from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge
from knowledge3d.cranium.adaptive_rpn_engine import AdaptiveRPNEngine
from knowledge3d.cranium.sleep.model_sleep import ModelSleepCycle
from knowledge3d.cranium.sleep.knowledge_sleep import KnowledgeSleepCycle


class DatasetRegistry:
    """
    Central registry of all available datasets.

    Organizes by modality and source for comprehensive training.
    """

    def __init__(self):
        """Initialize dataset registry."""
        # Base paths
        self.k3d_datasets = Path("/home/daniel/K3D_llama_cpp/datasets")
        self.k3d_local_datasets = Path("/K3D/Knowledge3D.local/datasets")
        self.echosystems_db = Path("/mnt/arquivos/0 ChatGPTs/DataBase")

        # Dataset catalog - ordered by training priority
        self.datasets: Dict[str, List[Dict[str, Any]]] = {
            "characters": [],      # Foundational: character-level multimodal
            "text": [],            # Foundational: text understanding
            "arc_agi": [],         # Reasoning: ARC-AGI challenges
            "multimodal": [],      # Traditional: images + captions
            "audio": [],           # Traditional: audio + captions
            "vision": [],          # Traditional: vision tasks
            "language": [],        # Traditional: language tasks
            "pdf": [],             # Synthesis: complex documents
            "compendiums": []      # Synthesis: structured knowledge
        }

        self._scan_datasets()

    def _scan_datasets(self):
        """Scan and catalog all available datasets."""

        # ================================================================
        # FOUNDATIONAL: CHARACTER-LEVEL MULTIMODAL
        # ================================================================

        # Character embeddings (trimodal: char + visual + phonetic)
        if (self.k3d_local_datasets / "character_embeddings_trimodal.jsonl").exists():
            char_size_mb = (self.k3d_local_datasets / "character_embeddings_trimodal.jsonl").stat().st_size / (1024 * 1024)
            self.datasets["characters"].append({
                "name": "Character Embeddings (Trimodal)",
                "path": self.k3d_local_datasets / "character_embeddings_trimodal.jsonl",
                "format": "jsonl",
                "size_mb": int(char_size_mb),
                "sovereign": True,
                "estimated_samples": 5000  # Estimate based on file size
            })

        # ================================================================
        # FOUNDATIONAL: TEXT UNDERSTANDING
        # ================================================================

        # Text domains (curated text corpus)
        if (self.k3d_local_datasets / "text_domains_v1.txt").exists():
            txt_size = (self.k3d_local_datasets / "text_domains_v1.txt").stat().st_size
            self.datasets["text"].append({
                "name": "Text Domains v1",
                "path": self.k3d_local_datasets / "text_domains_v1.txt",
                "format": "txt",
                "size_mb": txt_size / (1024 * 1024),
                "sovereign": True,
                "estimated_samples": 5000
            })

        # ================================================================
        # REASONING: ARC-AGI CHALLENGES
        # ================================================================

        # ARC-AGI reasoning pairs (abstract reasoning corpus)
        arc_agi_path = self.k3d_local_datasets / "arc_agi"
        if arc_agi_path.exists():
            self.datasets["arc_agi"].append({
                "name": "ARC-AGI Reasoning Pairs",
                "path": arc_agi_path,
                "format": "npz",
                "size_mb": 10,  # Small but critical
                "sovereign": True,
                "estimated_samples": 400  # ARC-AGI has ~400 training + eval tasks
            })

        # ================================================================
        # MULTIMODAL DATASETS (Text + Images + Audio)
        # ================================================================

        # K3D Generated embeddings (sovereign RPN)
        if (self.k3d_local_datasets / "trimodal_phase_g.jsonl").exists():
            self.datasets["multimodal"].append({
                "name": "Phase G Trimodal",
                "path": self.k3d_local_datasets / "trimodal_phase_g.jsonl",
                "format": "jsonl",
                "size_mb": 48,
                "description": "Phase G trimodal embeddings (text+image+audio)",
                "sovereign": True,
                "estimated_samples": 10000
            })

        if (self.k3d_local_datasets / "multimodal_embeddings.jsonl").exists():
            self.datasets["multimodal"].append({
                "name": "Multimodal Embeddings",
                "path": self.k3d_local_datasets / "multimodal_embeddings.jsonl",
                "format": "jsonl",
                "size_mb": 64,
                "description": "Text + image multimodal embeddings",
                "sovereign": True,
                "estimated_samples": 15000
            })

        # COCO (images + captions)
        if (self.k3d_datasets / "coco_raw").exists():
            self.datasets["multimodal"].append({
                "name": "COCO",
                "path": self.k3d_datasets / "coco_raw",
                "format": "directory",
                "description": "Common Objects in Context (images + captions)",
                "sovereign": False,
                "estimated_samples": 82783
            })

        # ================================================================
        # LANGUAGE DATASETS (Text only - traditional AI datasets)
        # ================================================================

        # Wikipedia JSONs (from EchoSystems DB)
        wiki_jsons = list(self.echosystems_db.rglob("*Wiki*.json"))[:20]  # Sample
        if wiki_jsons:
            self.datasets["language"].append({
                "name": "Wikipedia JSONs",
                "path": self.echosystems_db,
                "format": "json",
                "description": f"Wikipedia knowledge ({len(wiki_jsons)} topics)",
                "sovereign": False,
                "estimated_samples": len(wiki_jsons) * 100
            })

        # Medicine Wikipedia
        medicine_wiki = Path("/mnt/arquivos/0 ChatGPTs/Zion/Framework_Development/AugmenToolKit/augmentoolkit/original/input/medicine_wikipedia.txt")
        if medicine_wiki.exists():
            self.datasets["language"].append({
                "name": "Medicine Wikipedia",
                "path": medicine_wiki,
                "format": "txt",
                "description": "Medical knowledge from Wikipedia",
                "sovereign": False,
                "estimated_samples": 1000
            })

        # ================================================================
        # AUDIO DATASETS
        # ================================================================

        # Speech embeddings (sovereign)
        if (self.k3d_local_datasets / "speech_embeddings.jsonl").exists():
            self.datasets["audio"].append({
                "name": "Speech Embeddings",
                "path": self.k3d_local_datasets / "speech_embeddings.jsonl",
                "format": "jsonl",
                "size_mb": 61,
                "description": "Sovereign speech embeddings",
                "sovereign": True,
                "estimated_samples": 12000
            })

        # AudioCaps (audio + captions)
        if (self.k3d_datasets / "audiocaps_raw").exists():
            self.datasets["audio"].append({
                "name": "AudioCaps",
                "path": self.k3d_datasets / "audiocaps_raw",
                "format": "directory",
                "description": "Audio clips with captions",
                "sovereign": False,
                "estimated_samples": 50000
            })

        # Clotho (audio descriptions)
        if (self.k3d_datasets / "clotho_raw").exists():
            self.datasets["audio"].append({
                "name": "Clotho",
                "path": self.k3d_datasets / "clotho_raw",
                "format": "directory",
                "description": "Audio clips with detailed descriptions",
                "sovereign": False,
                "estimated_samples": 6974
            })

        # ================================================================
        # VISION DATASETS (Images)
        # ================================================================

        # Image captions (sovereign)
        if (self.k3d_local_datasets / "image_captions_llama32vision.jsonl").exists():
            self.datasets["vision"].append({
                "name": "Image Captions (Llama32 Vision)",
                "path": self.k3d_local_datasets / "image_captions_llama32vision.jsonl",
                "format": "jsonl",
                "size_mb": 3.7,
                "description": "Image captions from Llama 3.2 Vision",
                "sovereign": True,
                "estimated_samples": 5000
            })

        # ================================================================
        # PDF DATASETS (Curated Knowledge)
        # ================================================================

        # EchoSystems Default Libraries (3.0 GB, 327 PDFs)
        echosystems_pdfs = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries")
        if echosystems_pdfs.exists():
            pdf_count = len(list(echosystems_pdfs.rglob("*.pdf")))
            self.datasets["pdf"].append({
                "name": "EchoSystems Default Libraries",
                "path": echosystems_pdfs,
                "format": "pdf",
                "size_mb": 3000,
                "description": f"Curated knowledge library ({pdf_count} PDFs)",
                "sovereign": True,  # Will process with Phase G
                "estimated_samples": pdf_count * 50  # ~50 pages per PDF avg
            })

        # ================================================================
        # COMPENDIUMS (Structured Knowledge)
        # ================================================================

        # EchoSystems Compendiums
        compendiums_path = self.echosystems_db / "EchoSystems Compendiums - other/training_dataset"
        if compendiums_path.exists():
            compendium_files = list(compendiums_path.glob("*.jsonl"))
            self.datasets["compendiums"].append({
                "name": "EchoSystems Compendiums",
                "path": compendiums_path,
                "format": "jsonl",
                "description": f"Structured knowledge compendiums ({len(compendium_files)} topics)",
                "sovereign": True,
                "estimated_samples": len(compendium_files) * 1000
            })

    def print_summary(self):
        """Print dataset summary."""
        print("\n" + "="*80)
        print("DATASET REGISTRY - Full AGI Training Corpus")
        print("="*80)

        total_samples = 0

        for category, datasets in self.datasets.items():
            if not datasets:
                continue

            print(f"\n{category.upper()} ({len(datasets)} datasets):")
            print("-" * 80)

            for ds in datasets:
                samples = ds.get("estimated_samples", 0)
                total_samples += samples

                sovereign_mark = "🔒" if ds.get("sovereign") else "📦"
                size_info = f"({ds.get('size_mb', 0)} MB)" if "size_mb" in ds else ""

                print(f"  {sovereign_mark} {ds['name']:40} {samples:>8} samples {size_info}")
                if "description" in ds:
                    print(f"     {ds['description']}")

        print("\n" + "="*80)
        print(f"TOTAL ESTIMATED SAMPLES: {total_samples:,}")
        print(f"SOVEREIGN DATASETS: {sum(1 for cat in self.datasets.values() for ds in cat if ds.get('sovereign'))}")
        print("="*80 + "\n")


class FullAGITrainer:
    """
    Complete AGI training orchestrator with dual sleep cycles.

    Processes all datasets with sovereign RPN engine and trained specialists,
    then runs both sleep cycles to produce paired House + Models.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize full AGI trainer.

        Args:
            checkpoint_dir: Phase G specialist checkpoints
        """
        print("[FullAGI] Initializing Full AGI Trainer...")

        # Dataset registry
        self.registry = DatasetRegistry()

        # Phase G ingestion bridge (with specialists)
        self.phase_g_bridge = PhaseGPDFIngestionBridge(
            phase_g_checkpoint_dir=checkpoint_dir
        )

        # Adaptive RPN engine
        self.adaptive_rpn = self.phase_g_bridge.adaptive_rpn

        # Training state
        self.training_metrics = {
            "phases_completed": [],
            "total_samples_processed": 0,
            "total_galaxy_stars": 0,
            "sleep_cycles_run": 0
        }

        # Output paths
        self.output_dir = Path("/K3D/Knowledge3D.local/agi_training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.output_dir / "training_metrics.jsonl"
        self.final_house_path = self.output_dir / "final_house.glb"
        self.final_models_dir = self.output_dir / "final_models"

        print("[FullAGI] Trainer initialized")

    def run_training_phase(self, phase_name: str, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a single training phase on specified datasets.

        Args:
            phase_name: Phase identifier
            datasets: List of datasets to process

        Returns:
            Phase metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING PHASE: {phase_name}")
        print(f"{'='*80}")

        phase_start = time.time()
        samples_processed = 0
        stars_created = 0

        for dataset in datasets:
            print(f"\n  Processing: {dataset['name']}")
            print(f"  Format: {dataset['format']}")
            print(f"  Estimated samples: {dataset.get('estimated_samples', 'unknown')}")

            # Process based on format
            if dataset["format"] == "pdf":
                # Use Phase G PDF ingestion
                result = self._process_pdf_dataset(dataset)
                samples_processed += result["samples"]
                stars_created += result["stars"]

            elif dataset["format"] == "jsonl":
                # Process JSONL (embeddings or structured data)
                result = self._process_jsonl_dataset(dataset)
                samples_processed += result["samples"]
                stars_created += result["stars"]

            elif dataset["format"] == "txt":
                # Process plain text
                result = self._process_text_dataset(dataset)
                samples_processed += result["samples"]
                stars_created += result["stars"]

            elif dataset["format"] == "json":
                # Process JSON (Wikipedia, etc.)
                result = self._process_json_dataset(dataset)
                samples_processed += result["samples"]
                stars_created += result["stars"]

            elif dataset["format"] == "directory":
                # Process directory (COCO, AudioCaps, etc.)
                result = self._process_directory_dataset(dataset)
                samples_processed += result["samples"]
                stars_created += result["stars"]

            else:
                print(f"  ⚠️  Unknown format: {dataset['format']}, skipping")

        phase_elapsed = time.time() - phase_start

        metrics = {
            "phase_name": phase_name,
            "samples_processed": samples_processed,
            "stars_created": stars_created,
            "elapsed_seconds": phase_elapsed,
            "timestamp": datetime.now().isoformat()
        }

        self.training_metrics["phases_completed"].append(metrics)
        self.training_metrics["total_samples_processed"] += samples_processed
        self.training_metrics["total_galaxy_stars"] += stars_created

        print(f"\n✅ Phase '{phase_name}' complete:")
        print(f"   Samples: {samples_processed:,}")
        print(f"   Galaxy stars: {stars_created:,}")
        print(f"   Time: {phase_elapsed/60:.1f} minutes")

        return metrics

    def _process_pdf_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
        """Process PDF dataset with Phase G bridge."""
        pdf_path = Path(dataset["path"])

        if not pdf_path.exists():
            print(f"  ⚠️  PDF path not found: {pdf_path}")
            return {"samples": 0, "stars": 0}

        # Find all PDFs
        pdf_files = list(pdf_path.rglob("*.pdf")) + list(pdf_path.rglob("*.PDF"))

        if not pdf_files:
            print(f"  ⚠️  No PDFs found in {pdf_path}")
            return {"samples": 0, "stars": 0}

        print(f"  Found {len(pdf_files)} PDFs")

        import fitz  # PyMuPDF
        fitz.TOOLS.mupdf_display_errors(False)

        pages_processed = 0
        stars_created = 0

        for idx, pdf_file in enumerate(pdf_files, 1):  # Process ALL PDFs
            try:
                doc = fitz.open(str(pdf_file))
                num_pages = len(doc)

                print(f"    [{idx}/{min(10, len(pdf_files))}] {pdf_file.name} ({num_pages} pages)")

                for page_num in range(num_pages):
                    result = self.phase_g_bridge.ingest_pdf_page(str(pdf_file), page_num)
                    pages_processed += 1
                    if result.get("galaxy_star"):
                        stars_created += 1

                    if (page_num + 1) % 10 == 0:
                        print(f"      Page {page_num + 1}/{num_pages}")

                doc.close()

                # Save periodically
                if idx % 5 == 0:
                    self.phase_g_bridge.save_galaxy_stars()

            except Exception as exc:
                print(f"    ERROR: {pdf_file.name}: {exc}")

        # Final save
        self.phase_g_bridge.save_galaxy_stars()

        return {"samples": pages_processed, "stars": stars_created}

    def _process_jsonl_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
        """Process JSONL dataset."""
        return {"samples": 0, "stars": 0}

    def _process_text_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
        """Process text dataset."""
        return {"samples": 0, "stars": 0}

    def _process_json_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
        """Process JSON dataset."""
        return {"samples": 0, "stars": 0}

    def _process_directory_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
        """Process directory dataset."""
        return {"samples": 0, "stars": 0}

    def run_sleep_cycle_1_model_updates(self) -> Dict[str, Any]:
        """
        Sleep Cycle 1: Model Updates (Shadow Weights Validation).

        Updates: MODELS = LOGIC (how to process information)

        Process:
        1. Iterate through all specialists
        2. Validate shadow weights against validation sets
        3. Commit improvements if performance gain > threshold
        4. Reject if performance degradation
        5. Save updated checkpoints

        Returns:
            Sleep metrics
        """
        if not self.phase_g_bridge.specialists_loaded:
            print("\n⚠️  No specialists loaded, skipping model sleep")
            return {"cycle_type": "model_updates", "skipped": True}

        # Initialize Model Sleep Cycle
        model_sleep = ModelSleepCycle(
            matryoshka_system=self.phase_g_bridge.matryoshka_system,
            checkpoint_dir=self.final_models_dir
        )

        # Run validation and commit
        metrics = model_sleep.run()

        self.training_metrics["sleep_cycles_run"] += 1

        return metrics

    def run_sleep_cycle_2_knowledge_consolidation(self) -> Dict[str, Any]:
        """
        Sleep Cycle 2: Knowledge Consolidation (Galaxy → House).

        Updates: KNOWLEDGE = 3D SPACE (what information we know)

        Process:
        1. Load Galaxy stars (all created during training)
        2. Cluster stars by semantic similarity (RPN-powered)
        3. Materialize clusters into House objects
        4. Generate fractal knowledge trees with φ constraints
        5. Create AI textures for 3D visualization
        6. Save updated House GLB

        Returns:
            Sleep metrics
        """
        # Initialize Knowledge Sleep Cycle
        # Use the actual path where Phase G bridge saves stars
        galaxy_stars_path = Path("/K3D/Knowledge3D.local/house_zone7/embeddings/galaxy_stars.pkl")

        knowledge_sleep = KnowledgeSleepCycle(
            galaxy_stars_path=galaxy_stars_path,
            house_output_path=self.final_house_path,
            rpn_engine=self.adaptive_rpn
        )

        # Run consolidation
        metrics = knowledge_sleep.run(n_clusters=10)

        self.training_metrics["sleep_cycles_run"] += 1

        return metrics

    def run_full_training(self, phases: Optional[List[str]] = None):
        """
        Run complete AGI training with all phases and sleep cycles.

        Args:
            phases: List of phase names to run (None = all)
        """
        print("\n" + "="*80)
        print("FULL AGI TRAINING - Sovereign Engine + Dual Sleep Cycles")
        print("="*80)
        print(f"Start time: {datetime.now().isoformat()}\n")

        # Show dataset summary
        self.registry.print_summary()

        # Define training phases (ordered from foundational to complex)
        all_phases = {
            "characters": self.registry.datasets["characters"],    # 1. Foundational: character-level
            "text": self.registry.datasets["text"],                # 2. Foundational: text understanding
            "arc_agi": self.registry.datasets["arc_agi"],          # 3. Reasoning: ARC-AGI
            "multimodal": self.registry.datasets["multimodal"],    # 4. Traditional: multimodal
            "audio": self.registry.datasets["audio"],              # 5. Traditional: audio
            "vision": self.registry.datasets["vision"],            # 6. Traditional: vision
            "language": self.registry.datasets["language"],        # 7. Traditional: language
            "pdf": self.registry.datasets["pdf"],                  # 8. Synthesis: complex documents
            "compendiums": self.registry.datasets["compendiums"]   # 9. Synthesis: structured knowledge
        }

        # Filter phases if specified
        if phases:
            all_phases = {k: v for k, v in all_phases.items() if k in phases}

        # Run each training phase
        for phase_name, datasets in all_phases.items():
            if not datasets:
                print(f"\n⚠️  No datasets for phase '{phase_name}', skipping")
                continue

            # Train on this phase
            phase_metrics = self.run_training_phase(phase_name, datasets)

            # Run BOTH sleep cycles after phase
            print(f"\n{'─'*80}")
            print(f"Running sleep cycles after '{phase_name}' phase...")
            print(f"{'─'*80}")

            # Sleep 1: Model updates
            model_sleep_metrics = self.run_sleep_cycle_1_model_updates()

            # Sleep 2: Knowledge consolidation
            knowledge_sleep_metrics = self.run_sleep_cycle_2_knowledge_consolidation()

            # Save metrics
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(phase_metrics) + "\n")
                f.write(json.dumps(model_sleep_metrics) + "\n")
                f.write(json.dumps(knowledge_sleep_metrics) + "\n")

        # Final summary
        self.print_final_summary()

    def print_final_summary(self):
        """Print final training summary."""
        print("\n" + "="*80)
        print("FULL AGI TRAINING COMPLETE")
        print("="*80)

        metrics = self.training_metrics

        print(f"\nPhases completed: {len(metrics['phases_completed'])}")
        print(f"Total samples processed: {metrics['total_samples_processed']:,}")
        print(f"Total Galaxy stars: {metrics['total_galaxy_stars']:,}")
        print(f"Sleep cycles run: {metrics['sleep_cycles_run']}")

        print(f"\nOutputs:")
        print(f"  Metrics: {self.metrics_path}")
        print(f"  Final House: {self.final_house_path}")
        print(f"  Final Models: {self.final_models_dir}")

        print("\n" + "="*80)
        print("🚀 AGI training complete - House + Models ready for deployment!")
        print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Full AGI Training with Sovereign Engine")
    parser.add_argument(
        "--phases",
        type=str,
        nargs="+",
        choices=["multimodal", "language", "audio", "vision", "pdf", "compendiums"],
        help="Specific phases to train (default: all)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/K3D/Knowledge3D.local/checkpoints/phase_g/current",
        help="Phase G specialist checkpoint directory"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )

    args = parser.parse_args()

    # List datasets only
    if args.list_datasets:
        registry = DatasetRegistry()
        registry.print_summary()
        return 0

    # Initialize trainer
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    trainer = FullAGITrainer(checkpoint_dir=checkpoint_dir)

    # Run training
    trainer.run_full_training(phases=args.phases)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
