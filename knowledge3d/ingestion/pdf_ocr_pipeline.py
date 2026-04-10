"""Modular PDF OCR pipeline with GPU acceleration and Ollama integration.

This module orchestrates the complete PDF processing pipeline including:
- GPU-accelerated OCR for non-ready PDFs
- Persistent OCR layer creation
- Ollama-assisted second pass enrichment
- Modular architecture with kernel-based processing
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knowledge3d.ingestion.ocr_gpu_processor import GPUPDFOCRProcessor, OllamaSecondPassEnricher
from knowledge3d.ingestion.proceduralizer_wine_bridge import ProceduralizerWineBridge
from knowledge3d.ingestion.proceduralizer_contract import ProceduralizerRequest, ProceduralizerBundle


class PDFKnowledgePipeline:
    """Complete PDF processing pipeline with GPU OCR and Ollama enrichment."""
    
    def __init__(
        self,
        *,
        gpu_ocr_processor: GPUPDFOCRProcessor | None = None,
        ollama_enricher: OllamaSecondPassEnricher | None = None,
        proceduralizer_bridge: ProceduralizerWineBridge | None = None,
        enable_gpu_ocr: bool = True,
        enable_ollama_enrichment: bool = True,
        ocr_model: str = "deepseek-ocr:latest",
        enrichment_model: str = "kimi-k2-thinking:cloud"
    ):
        self.enable_gpu_ocr = enable_gpu_ocr
        self.enable_ollama_enrichment = enable_ollama_enrichment
        
        # Initialize components
        self.gpu_ocr_processor = gpu_ocr_processor or GPUPDFOCRProcessor(ocr_model=ocr_model)
        self.ollama_enricher = ollama_enricher or OllamaSecondPassEnricher(enrichment_model=enrichment_model)
        self.proceduralizer_bridge = proceduralizer_bridge or ProceduralizerWineBridge()
        
        self.logger = logging.getLogger(__name__)
        
    def process_pdf(
        self,
        pdf_path: Path,
        *,
        output_dir: Path,
        domain_hint: str = "General",
        max_pages: int = 0,
        ocr_options: Dict[str, Any] | None = None,
        enrichment_options: Dict[str, Any] | None = None,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """Process PDF through complete pipeline with OCR and enrichment."""
        
        self.logger.info(f"Starting PDF processing pipeline for: {pdf_path}")
        
        results = {
            "pdf_path": str(pdf_path),
            "pipeline_version": "2.0",
            "stages": {},
            "total_pages": 0,
            "processed_pages": 0,
            "ocr_pages": 0,
            "enriched_pages": 0,
            "errors": []
        }
        
        try:
            # Stage 1: PDF Analysis and OCR Detection
            self.logger.info("Stage 1: PDF Analysis and OCR Detection")
            pdf_analysis = self._analyze_pdf_needs_ocr(pdf_path)
            results["stages"]["pdf_analysis"] = pdf_analysis
            
            # Stage 2: GPU-Accelerated OCR (if needed)
            ocr_results = {}
            if self.enable_gpu_ocr and pdf_analysis["needs_ocr"]:
                self.logger.info("Stage 2: GPU-Accelerated OCR Processing")
                ocr_output_path = output_dir / f"{pdf_path.stem}_with_ocr.pdf" if save_intermediate else None
                
                ocr_results = self.gpu_ocr_processor.process_pdf_with_gpu_ocr(
                    pdf_path,
                    output_path=ocr_output_path,
                    page_range=pdf_analysis["ocr_pages"],
                    ocr_options=ocr_options,
                    save_ocr_layer=True
                )
                
                results["stages"]["ocr_processing"] = {
                    "pages_processed": len(ocr_results),
                    "ocr_output_path": str(ocr_output_path) if ocr_output_path else None,
                    "gpu_memory_used": self._estimate_gpu_memory_usage(ocr_results)
                }
                results["ocr_pages"] = len(ocr_results)
                
                # Use OCR-enhanced PDF for further processing
                if ocr_output_path and ocr_output_path.exists():
                    pdf_path = ocr_output_path
            
            # Stage 3: First Pass Proceduralization
            self.logger.info("Stage 3: First Pass Proceduralization")
            first_pass_results = self._perform_first_pass_proceduralization(
                pdf_path, domain_hint, max_pages
            )
            results["stages"]["first_pass"] = first_pass_results
            results["processed_pages"] = first_pass_results.get("pages_processed", 0)
            
            # Stage 4: Ollama-Assisted Second Pass Enrichment
            if self.enable_ollama_enrichment and first_pass_results.get("knowledge_packets"):
                self.logger.info("Stage 4: Ollama-Assisted Second Pass Enrichment")
                
                enriched_results = self._perform_ollama_second_pass(
                    pdf_path,
                    first_pass_results,
                    ocr_results,
                    enrichment_options
                )
                
                results["stages"]["second_pass"] = enriched_results
                results["enriched_pages"] = enriched_results.get("packets_enriched", 0)
            
            # Stage 5: Final Payload Generation
            self.logger.info("Stage 5: Final Payload Generation")
            final_payload = self._generate_final_payload(results)
            results["final_payload"] = final_payload
            
            self.logger.info(f"PDF pipeline completed successfully. "
                           f"Pages: {results['processed_pages']}, "
                           f"OCR: {results['ocr_pages']}, "
                           f"Enriched: {results['enriched_pages']}")
            
        except Exception as e:
            self.logger.error(f"PDF pipeline failed: {e}")
            results["errors"].append(str(e))
            raise
        
        return results
    
    def _analyze_pdf_needs_ocr(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze PDF to determine if OCR is needed and which pages."""
        
        import fitz  # PyMuPDF
        
        needs_ocr = False
        ocr_pages = []
        text_pages = []
        total_pages = 0
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    text = page.get_text("text").strip()
                    
                    # Simple heuristic: if page has very little text, it might need OCR
                    if len(text) < 50:  # Less than 50 characters
                        # Check if there are images that might contain text
                        images = page.get_images()
                        if images:
                            needs_ocr = True
                            ocr_pages.append(page_num)
                        else:
                            text_pages.append(page_num)
                    else:
                        text_pages.append(page_num)
                        
        except Exception as e:
            self.logger.warning(f"PDF analysis failed, assuming OCR needed: {e}")
            needs_ocr = True
            ocr_pages = list(range(total_pages))
        
        return {
            "needs_ocr": needs_ocr,
            "ocr_pages": ocr_pages,
            "text_pages": text_pages,
            "total_pages": total_pages,
            "analysis_method": "text_length_heuristic"
        }
    
    def _perform_first_pass_proceduralization(
        self,
        pdf_path: Path,
        domain_hint: str,
        max_pages: int
    ) -> Dict[str, Any]:
        """Perform first pass proceduralization on PDF content."""
        
        import fitz  # PyMuPDF
        
        results = {
            "pages_processed": 0,
            "knowledge_packets": [],
            "processing_time": 0,
            "errors": []
        }
        
        start_time = self._get_timestamp()
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                limit = total_pages if max_pages <= 0 else min(total_pages, max_pages)
                
                for page_num in range(limit):
                    try:
                        page = doc.load_page(page_num)
                        text_content = page.get_text("text").strip()
                        
                        if not text_content:
                            continue
                        
                        # Create proceduralizer request
                        request = ProceduralizerRequest(
                            source_kind="pdf_page",
                            source_id=f"{pdf_path.stem}_page_{page_num + 1}",
                            source_path=str(pdf_path),
                            domain_hint=domain_hint,
                            content=text_content,
                            context_chunks=[],
                            existing_ref_menu="",
                            quality_profile="quality",
                            ingest_mode="augment"
                        )
                        
                        # Submit to proceduralizer
                        receipt = self.proceduralizer_bridge.submit(request)
                        
                        if receipt_is_usable(receipt):
                            results["knowledge_packets"].extend(
                                receipt.parsed_bundle.get("knowledge_packets", [])
                            )
                            results["pages_processed"] += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to process page {page_num + 1}: {e}")
                        results["errors"].append(f"Page {page_num + 1}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"First pass proceduralization failed: {e}")
            results["errors"].append(str(e))
            raise
        
        results["processing_time"] = self._get_timestamp() - start_time
        return results
    
    def _perform_ollama_second_pass(
        self,
        pdf_path: Path,
        first_pass_results: Dict[str, Any],
        ocr_results: Dict[int, str],
        enrichment_options: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        """Perform Ollama-assisted second pass enrichment."""
        
        results = {
            "packets_enriched": 0,
            "enriched_packets": [],
            "processing_time": 0,
            "errors": []
        }
        
        start_time = self._get_timestamp()
        
        try:
            knowledge_packets = first_pass_results.get("knowledge_packets", [])
            
            for packet in knowledge_packets:
                try:
                    # Extract original text and context
                    original_text = packet.get("summary", "")
                    context_chunks = self._build_context_chunks(packet, ocr_results)
                    
                    # Perform Ollama enrichment
                    enriched_data = self.ollama_enricher.enrich_with_ollama(
                        original_text=original_text,
                        first_pass_results=packet,
                        context_chunks=context_chunks,
                        enrichment_options=enrichment_options
                    )
                    
                    if "error" not in enriched_data:
                        results["enriched_packets"].append(enriched_data)
                        results["packets_enriched"] += 1
                    else:
                        results["errors"].append(f"Enrichment error: {enriched_data['error']}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to enrich packet: {e}")
                    results["errors"].append(str(e))
                    
        except Exception as e:
            self.logger.error(f"Second pass enrichment failed: {e}")
            results["errors"].append(str(e))
            raise
        
        results["processing_time"] = self._get_timestamp() - start_time
        return results
    
    def _build_context_chunks(self, packet: Dict[str, Any], ocr_results: Dict[int, str]) -> List[str]:
        """Build context chunks for enrichment from packet and OCR results."""
        
        context_chunks = []
        
        # Add packet relationships as context
        relationships = packet.get("relationships", {})
        if relationships:
            context_chunks.append(f"Relationships: {json.dumps(relationships)}")
        
        # Add OCR context if available
        if ocr_results:
            ocr_context = "OCR extracted text from images: " + " ".join(
                text for text in ocr_results.values() if text.strip()
            )
            context_chunks.append(ocr_context[:1000])  # Limit length
        
        # Add taxonomy and reality refs as context
        taxonomy_refs = packet.get("taxonomy_refs", [])
        reality_refs = packet.get("reality_refs", [])
        
        if taxonomy_refs:
            context_chunks.append(f"Taxonomy context: {', '.join(taxonomy_refs[:10])}")
        
        if reality_refs:
            context_chunks.append(f"Reality context: {', '.join(reality_refs[:10])}")
        
        return context_chunks
    
    def _generate_final_payload(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final payload from pipeline results."""
        
        enriched_packets = []
        
        # Collect enriched packets from second pass
        second_pass_results = pipeline_results["stages"].get("second_pass", {})
        if second_pass_results.get("enriched_packets"):
            enriched_packets.extend(second_pass_results["enriched_packets"])
        
        # Fallback to first pass packets if no enrichment
        if not enriched_packets:
            first_pass_results = pipeline_results["stages"].get("first_pass", {})
            enriched_packets = first_pass_results.get("knowledge_packets", [])
        
        return {
            "source_pdf": pipeline_results["pdf_path"],
            "processing_timestamp": self._get_timestamp(),
            "pipeline_version": pipeline_results["pipeline_version"],
            "total_pages": pipeline_results["total_pages"],
            "processed_pages": pipeline_results["processed_pages"],
            "knowledge_packets": enriched_packets,
            "metadata": {
                "ocr_pages": pipeline_results["ocr_pages"],
                "enriched_pages": pipeline_results["enriched_pages"],
                "gpu_acceleration": self.enable_gpu_ocr,
                "ollama_enrichment": self.enable_ollama_enrichment
            }
        }
    
    def _estimate_gpu_memory_usage(self, ocr_results: Dict[int, str]) -> int:
        """Estimate GPU memory usage in bytes."""
        
        # Rough estimate: 4MB per page for image processing + overhead
        return len(ocr_results) * 4 * 1024 * 1024 + 100 * 1024 * 1024  # 100MB overhead
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        
        import time
        return time.time()


class ModularPDFProcessor:
    """Modular PDF processor that can be configured for different processing modes."""
    
    def __init__(self, *, config: Dict[str, Any] | None = None):
        self.config = config or self._get_default_config()
        
        # Initialize pipeline components based on config
        self.pipeline = PDFKnowledgePipeline(
            enable_gpu_ocr=self.config.get("enable_gpu_ocr", True),
            enable_ollama_enrichment=self.config.get("enable_ollama_enrichment", True),
            ocr_model=self.config.get("ocr_model", "deepseek-ocr:latest"),
            enrichment_model=self.config.get("enrichment_model", "kimi-k2-thinking:cloud")
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the processor."""
        
        return {
            "enable_gpu_ocr": True,
            "enable_ollama_enrichment": True,
            "ocr_model": "deepseek-ocr:latest",
            "enrichment_model": "kimi-k2-thinking:cloud",
            "gpu_memory_limit_gb": 10,  # Stay within RTX 3060 12GB limit
            "max_concurrent_pages": 4,
            "ocr_options": {
                "contrast": 1.2,
                "brightness": 10.0,
                "gamma": 0.9,
                "text_threshold": 50.0
            },
            "enrichment_options": {
                "depth_level": "comprehensive",
                "max_context_length": 8000
            }
        }
    
    def process_single_pdf(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single PDF through the complete pipeline."""
        
        return self.pipeline.process_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            ocr_options=self.config.get("ocr_options"),
            enrichment_options=self.config.get("enrichment_options"),
            save_intermediate=True
        )
    
    def process_pdf_batch(
        self,
        pdf_paths: List[Path],
        output_dir: Path,
        *,
        resume_from: int = 0
    ) -> List[Dict[str, Any]]:
        """Process multiple PDFs in batch with resume capability."""
        
        results = []
        
        for i, pdf_path in enumerate(pdf_paths):
            if i < resume_from:
                continue
                
            try:
                result = self.process_single_pdf(pdf_path, output_dir / f"batch_{i}")
                results.append(result)
                
            except Exception as e:
                # Log error but continue with next PDF
                logging.error(f"Failed to process {pdf_path}: {e}")
                results.append({
                    "pdf_path": str(pdf_path),
                    "error": str(e),
                    "stage": "batch_processing"
                })
        
        return results


# Export main classes
__all__ = [
    "PDFKnowledgePipeline",
    "ModularPDFProcessor",
    "GPUPDFOCRProcessor",
    "OllamaSecondPassEnricher"
]