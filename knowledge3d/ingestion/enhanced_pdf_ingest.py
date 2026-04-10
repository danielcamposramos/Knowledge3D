#!/usr/bin/env python3
"""Enhanced PDF ingestion with GPU OCR and Ollama second pass enrichment.

This script replaces the fundamental PDF ingestion with a modular, kernel-based
pipeline that includes:
- GPU-accelerated OCR for non-ready PDFs
- Persistent OCR layer creation
- Ollama-assisted second pass semantic enrichment
- Modular architecture with multiple processing stages
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from knowledge3d.ingestion.pdf_ocr_pipeline import ModularPDFProcessor
from knowledge3d.ingestion.ocr_gpu_processor import GPUPDFOCRProcessor, OllamaSecondPassEnricher


def setup_logging(*, verbose: bool = False) -> None:
    """Configure logging for the ingestion pipeline."""
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_pdf_list(pdf_list_path: Path) -> list[Path]:
    """Load PDF paths from a list file."""
    
    pdf_paths = []
    
    if not pdf_list_path.exists():
        logging.error(f"PDF list file not found: {pdf_list_path}")
        return pdf_paths
    
    try:
        with open(pdf_list_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                pdf_path = Path(line)
                if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
                    pdf_paths.append(pdf_path)
                else:
                    logging.warning(f"Line {line_num}: Invalid PDF path: {line}")
                    
    except Exception as e:
        logging.error(f"Failed to load PDF list: {e}")
        
    return pdf_paths


def analyze_ocr_needs(pdf_paths: list[Path]) -> dict[str, Any]:
    """Analyze which PDFs need OCR processing."""
    
    analysis = {
        "total_pdfs": len(pdf_paths),
        "ocr_needed": 0,
        "text_ready": 0,
        "ocr_candidates": [],
        "text_ready_pdfs": []
    }
    
    for pdf_path in pdf_paths:
        try:
            import fitz  # PyMuPDF
            
            with fitz.open(pdf_path) as doc:
                # Simple heuristic: check if PDF has extractable text
                text_pages = 0
                image_pages = 0
                
                for page in doc:
                    text = page.get_text("text").strip()
                    images = page.get_images()
                    
                    if len(text) > 100:  # Substantial text content
                        text_pages += 1
                    elif images:
                        image_pages += 1
                
                # Decision logic
                if image_pages > text_pages or text_pages == 0:
                    analysis["ocr_needed"] += 1
                    analysis["ocr_candidates"].append(str(pdf_path))
                else:
                    analysis["text_ready"] += 1
                    analysis["text_ready_pdfs"].append(str(pdf_path))
                    
        except Exception as e:
            logging.warning(f"Failed to analyze {pdf_path}: {e}")
            # Default to OCR needed if analysis fails
            analysis["ocr_needed"] += 1
            analysis["ocr_candidates"].append(str(pdf_path))
    
    return analysis


def create_processor_config(args: argparse.Namespace) -> dict[str, Any]:
    """Create configuration for the modular PDF processor."""
    
    config = {
        "enable_gpu_ocr": not args.skip_ocr,
        "enable_ollama_enrichment": not args.skip_enrichment,
        "ocr_model": args.ocr_model,
        "enrichment_model": args.enrichment_model,
        "gpu_memory_limit_gb": args.gpu_memory_limit,
        "max_concurrent_pages": args.max_concurrent_pages,
        "ocr_options": {
            "contrast": args.ocr_contrast,
            "brightness": args.ocr_brightness,
            "gamma": args.ocr_gamma,
            "text_threshold": args.ocr_threshold
        },
        "enrichment_options": {
            "depth_level": args.enrichment_depth,
            "max_context_length": args.max_context_length
        }
    }
    
    return config


def process_pdf_with_enhanced_pipeline(
    pdf_path: Path,
    output_dir: Path,
    *,
    config: dict[str, Any],
    resume_from: int = 0
) -> dict[str, Any]:
    """Process a single PDF through the enhanced pipeline."""
    
    try:
        logging.info(f"Processing PDF: {pdf_path}")
        
        # Initialize processor with config
        processor = ModularPDFProcessor(config=config)
        
        # Create output directory for this PDF
        pdf_output_dir = output_dir / pdf_path.stem
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the PDF
        result = processor.process_single_pdf(pdf_path, pdf_output_dir)
        
        # Save detailed results
        result_file = pdf_output_dir / "processing_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Completed processing: {pdf_path}")
        logging.info(f"Results saved to: {result_file}")
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
        return {
            "pdf_path": str(pdf_path),
            "error": str(e),
            "stage": "pipeline_processing"
        }


def main() -> int:
    """Main entry point for enhanced PDF ingestion."""
    
    parser = argparse.ArgumentParser(
        description="Enhanced PDF ingestion with GPU OCR and Ollama enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF with default settings
  python enhanced_pdf_ingest.py --pdf /path/to/document.pdf --output /output/dir
  
  # Process PDF list with OCR and enrichment
  python enhanced_pdf_ingest.py --pdf-list pdfs.txt --output /output/dir --verbose
  
  # Process with custom OCR settings
  python enhanced_pdf_ingest.py --pdf document.pdf --output /output --ocr-contrast 1.5 --ocr-brightness 15
  
  # Process with specific models
  python enhanced_pdf_ingest.py --pdf-list pdfs.txt --output /output --ocr-model deepseek-ocr:latest --enrichment-model kimi-k2-thinking:cloud
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf", type=Path, help="Single PDF file to process")
    input_group.add_argument("--pdf-list", type=Path, help="File containing list of PDF paths")
    
    # Output options
    parser.add_argument("--output", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--domain-hint", default="General", help="Domain hint for proceduralization")
    
    # OCR options
    ocr_group = parser.add_argument_group("OCR Configuration")
    ocr_group.add_argument("--skip-ocr", action="store_true", help="Skip OCR processing")
    ocr_group.add_argument("--ocr-model", default="deepseek-ocr:latest", help="Ollama model for OCR")
    ocr_group.add_argument("--ocr-contrast", type=float, default=1.2, help="OCR image contrast enhancement")
    ocr_group.add_argument("--ocr-brightness", type=float, default=10.0, help="OCR image brightness enhancement")
    ocr_group.add_argument("--ocr-gamma", type=float, default=0.9, help="OCR image gamma correction")
    ocr_group.add_argument("--ocr-threshold", type=float, default=50.0, help="OCR text detection threshold")
    
    # Enrichment options
    enrich_group = parser.add_argument_group("Enrichment Configuration")
    enrich_group.add_argument("--skip-enrichment", action="store_true", help="Skip Ollama second pass enrichment")
    enrich_group.add_argument("--enrichment-model", default="kimi-k2-thinking:cloud", help="Ollama model for enrichment")
    enrich_group.add_argument("--enrichment-depth", default="comprehensive", choices=["basic", "comprehensive", "deep"], help="Enrichment depth level")
    enrich_group.add_argument("--max-context-length", type=int, default=8000, help="Maximum context length for enrichment")
    
    # Performance options
    perf_group = parser.add_argument_group("Performance Configuration")
    perf_group.add_argument("--gpu-memory-limit", type=int, default=10, help="GPU memory limit in GB (RTX 3060: 12GB)")
    perf_group.add_argument("--max-concurrent-pages", type=int, default=4, help="Maximum concurrent pages to process")
    perf_group.add_argument("--max-pages", type=int, default=0, help="Maximum pages to process per PDF (0 = all)")
    
    # Other options
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from PDF index in batch")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze OCR needs, don't process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    try:
        # Determine PDF paths to process
        if args.pdf:
            pdf_paths = [args.pdf]
        else:
            pdf_paths = load_pdf_list(args.pdf_list)
        
        if not pdf_paths:
            logging.error("No valid PDFs found to process")
            return 1
        
        logging.info(f"Found {len(pdf_paths)} PDFs to process")
        
        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Analyze OCR needs if requested
        if args.analyze_only:
            logging.info("Performing OCR analysis only...")
            ocr_analysis = analyze_ocr_needs(pdf_paths)
            
            analysis_file = args.output / "ocr_analysis.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(ocr_analysis, f, indent=2, ensure_ascii=False)
            
            logging.info(f"OCR analysis saved to: {analysis_file}")
            logging.info(f"OCR needed: {ocr_analysis['ocr_needed']}, Text ready: {ocr_analysis['text_ready']}")
            return 0
        
        # Create processor configuration
        config = create_processor_config(args)
        
        # Process PDFs
        logging.info("Starting enhanced PDF processing pipeline...")
        
        total_results = []
        for i, pdf_path in enumerate(pdf_paths):
            if i < args.resume_from:
                continue
            
            try:
                result = process_pdf_with_enhanced_pipeline(
                    pdf_path, args.output, config=config, resume_from=i
                )
                total_results.append(result)
                
                # Log progress
                if result.get("error"):
                    logging.error(f"Failed: {pdf_path}")
                else:
                    logging.info(f"Success: {pdf_path} "
                               f"(Pages: {result.get('processed_pages', 0)}, "
                               f"OCR: {result.get('ocr_pages', 0)}, "
                               f"Enriched: {result.get('enriched_pages', 0)})")
                
            except Exception as e:
                logging.error(f"Critical failure on {pdf_path}: {e}")
                total_results.append({
                    "pdf_path": str(pdf_path),
                    "error": str(e),
                    "stage": "main_processing"
                })
        
        # Save summary results
        summary_file = args.output / "processing_summary.json"
        summary = {
            "total_pdfs": len(pdf_paths),
            "processed_pdfs": len(total_results),
            "successful_pdfs": len([r for r in total_results if not r.get("error")]),
            "failed_pdfs": len([r for r in total_results if r.get("error")]),
            "results": total_results,
            "config": config
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Processing complete. Summary saved to: {summary_file}")
        logging.info(f"Success rate: {summary['successful_pdfs']}/{summary['total_pdfs']}")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        return 130
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())