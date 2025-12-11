#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup foundational PDFs directory structure.

Run this script to:
1. Create the expected directory structure at /K3D/Knowledge3D.local/datasets/foundational_pdfs/
2. Symlink (or copy) your actual PDFs into the correct category folders

Configure PDF_SOURCES below with your actual PDF locations.
"""

from pathlib import Path
import shutil
import os

# Target base directory (where ingestion script expects PDFs)
TARGET_BASE = Path("/K3D/Knowledge3D.local/datasets/foundational_pdfs")

# Expected category structure (from FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
CATEGORIES = {
    "Advanced Mathematics/": [],
    "Pedagogy & Learning/": [],
    "Language, Grammar & Semantics/": [],
    "Eloquence, Rhetoric & Persuasion/": [],
    "Self-Reflection/": [],
    "Story Telling/": [],
    "Acting - Delivery/": [],
    "Context & Contextual Understanding/": [],
    "Temporal Understanding/": [],
    "Academic Research Methods/": [],
}

# ============================================================================
# CONFIGURE YOUR PDF SOURCES HERE
# Each entry: "Category Name/": [list of source paths to PDFs or directories]
#
# Examples:
#   "Advanced Mathematics/": [
#       "/path/to/calculus_book.pdf",
#       "/path/to/math_folder/",  # All PDFs in this folder
#   ],
# ============================================================================
PDF_SOURCES = {
    "Advanced Mathematics/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/",
    ],
    "Pedagogy & Learning/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Teach/",
    ],
    "Language, Grammar & Semantics/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Erudition/",
    ],
    "Eloquence, Rhetoric & Persuasion/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Eloquence/",
    ],
    "Self-Reflection/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Self Reflection/",
    ],
    "Story Telling/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Story Telling/",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Story Telling/Source/",
    ],
    "Acting - Delivery/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Acting/",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Acting/Source/",
    ],
    "Context & Contextual Understanding/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Context/",
    ],
    "Temporal Understanding/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Understand Time/",
    ],
    "Academic Research Methods/": [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Academic Research/",
    ],
}


def create_directory_structure():
    """Create the expected directory structure."""
    print(f"Creating directory structure at: {TARGET_BASE}")
    TARGET_BASE.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES:
        cat_path = TARGET_BASE / category
        cat_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {cat_path}")


def link_pdfs(use_symlinks: bool = True):
    """Link or copy PDFs from sources to target directories."""
    total_linked = 0

    for category, sources in PDF_SOURCES.items():
        if not sources:
            continue

        target_dir = TARGET_BASE / category
        print(f"\nProcessing {category}:")

        for source in sources:
            source_path = Path(source)

            if not source_path.exists():
                print(f"  [WARN] Source not found: {source_path}")
                continue

            # If source is a directory, link all PDFs inside
            if source_path.is_dir():
                pdf_files = list(source_path.glob("*.pdf"))
            else:
                pdf_files = [source_path] if source_path.suffix.lower() == ".pdf" else []

            for pdf in pdf_files:
                target = target_dir / pdf.name

                if target.exists():
                    print(f"  [SKIP] Already exists: {pdf.name}")
                    continue

                if use_symlinks:
                    target.symlink_to(pdf.resolve())
                    print(f"  [LINK] {pdf.name}")
                else:
                    shutil.copy2(pdf, target)
                    print(f"  [COPY] {pdf.name}")

                total_linked += 1

    return total_linked


def show_status():
    """Show current status of PDF directories."""
    print("\n=== Current PDF Status ===")

    if not TARGET_BASE.exists():
        print(f"Target base does not exist: {TARGET_BASE}")
        return

    total = 0
    for category in CATEGORIES:
        cat_path = TARGET_BASE / category
        if cat_path.exists():
            pdfs = list(cat_path.glob("*.pdf"))
            count = len(pdfs)
            total += count
            status = f"{count} PDFs" if count else "(empty)"
            print(f"  {category}: {status}")
        else:
            print(f"  {category}: (not created)")

    print(f"\nTotal: {total} PDFs")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup foundational PDFs directory")
    parser.add_argument("--create", action="store_true", help="Create directory structure")
    parser.add_argument("--link", action="store_true", help="Link PDFs from configured sources")
    parser.add_argument("--copy", action="store_true", help="Copy PDFs instead of symlinking")
    parser.add_argument("--status", action="store_true", help="Show current status")

    args = parser.parse_args()

    if args.create:
        create_directory_structure()

    if args.link or args.copy:
        count = link_pdfs(use_symlinks=not args.copy)
        print(f"\nLinked/copied {count} PDFs")

    if args.status or not any([args.create, args.link, args.copy]):
        show_status()

    if not any([args.create, args.link, args.copy, args.status]):
        print("\nUsage:")
        print("  1. Edit PDF_SOURCES in this script with your PDF paths")
        print("  2. Run: python scripts/setup_foundational_pdfs.py --create --link")
        print("  3. Check: python scripts/setup_foundational_pdfs.py --status")


if __name__ == "__main__":
    main()
