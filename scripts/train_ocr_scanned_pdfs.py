#!/usr/bin/env python3
"""
Train OCR on ONLY scanned PDFs with OCR layers.
These are the best training data because they have:
- Ground truth text (OCR layer)
- Actual character images (scanned pixels)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_full_agi_sovereign import AGITrainer


def main():
    """Train OCR on only scanned PDFs with OCR layers."""

    # The 6 scanned PDFs with OCR layers (from analysis)
    scanned_pdfs = [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Calculus/Advanced_Calculus.pdf",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Numerology/The Greek Qabalah_ Alphabetic Mysticism and Numerology in the Ancient World-State University of New York Press (2003).pdf",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Context/ssoar-1987-iversen-introduction_to_contextual_analysis.pdf",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/19990053708.pdf",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/19730016146.pdf",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/RPG Design/LA1-Cover.pdf",
    ]

    # Verify all PDFs exist
    valid_pdfs = []
    for pdf_path in scanned_pdfs:
        p = Path(pdf_path)
        if p.exists():
            valid_pdfs.append(p)
            print(f"✓ Found: {p.name}")
        else:
            print(f"⚠️  Missing: {p.name}")

    if not valid_pdfs:
        print("\n❌ No scanned PDFs found!")
        return 1

    print(f"\n📊 Training on {len(valid_pdfs)} scanned PDFs with OCR layers")

    # Count total pages
    import fitz
    total_pages = 0
    for pdf_path in valid_pdfs:
        try:
            doc = fitz.open(pdf_path)
            total_pages += len(doc)
            doc.close()
        except:
            pass

    print(f"   Total pages: {total_pages}")
    print()

    # Create trainer
    trainer = AGITrainer()

    # Override PDF dataset to use only scanned PDFs
    trainer.training_config["phases"]["pdf"]["datasets"] = [
        {
            "name": f"Scanned PDF: {pdf_path.stem}",
            "type": "pdf",
            "path": str(pdf_path),
            "format": "pdf"
        }
        for pdf_path in valid_pdfs
    ]

    # Run PDF training
    print("=" * 80)
    print("STARTING OCR TRAINING - SCANNED PDFs ONLY")
    print("=" * 80)
    print()

    trainer.run_training_phase("pdf", trainer.training_config["phases"]["pdf"]["datasets"])

    # Run sleep cycles
    print()
    print("=" * 80)
    print("RUNNING SLEEP CYCLES")
    print("=" * 80)
    trainer.run_sleep_cycles("pdf")

    print()
    print("=" * 80)
    print("OCR TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Test APOLLO.PDF with new weights:")
    print("   python scripts/test_apollo_ground_truth.py")
    print()
    print("2. Check F1 score improvement (should be >>0%)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
