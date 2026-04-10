"""GPU-accelerated OCR processing for PDF documents using CUDA kernels.

This module provides high-performance OCR processing for PDFs that need text extraction,
with GPU kernel acceleration and persistent OCR layer creation.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import pypdfium2 as pdfium
from PIL import Image

from knowledge3d.bridge.gpu_context import GPUContext
from knowledge3d.utils.cuda_kernels import load_cuda_kernel


class GPUPDFOCRProcessor:
    """GPU-accelerated PDF OCR processor with persistent layer creation."""
    
    def __init__(self, *, gpu_context: GPUContext | None = None, ocr_model: str = "deepseek-ocr:latest"):
        self.gpu_context = gpu_context or GPUContext()
        self.ocr_model = ocr_model
        self.ocr_kernel = self._load_ocr_kernel()
        
    def _load_ocr_kernel(self) -> Any:
        """Load CUDA kernel for GPU-accelerated OCR preprocessing."""
        kernel_source = """
        // GPU kernel for image preprocessing before OCR
        extern "C" {
        
        __global__ void preprocess_image_for_ocr(
            const unsigned char* input,
            unsigned char* output,
            int width,
            int height,
            float contrast,
            float brightness,
            float gamma
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (idx < width && idy < height) {
                int pixel_idx = idy * width + idx;
                float pixel = input[pixel_idx];
                
                // Apply contrast and brightness adjustments
                pixel = (pixel - 128.0f) * contrast + 128.0f + brightness;
                pixel = fmaxf(0.0f, fminf(255.0f, pixel));
                
                // Apply gamma correction
                pixel = 255.0f * powf(pixel / 255.0f, gamma);
                
                output[pixel_idx] = (unsigned char)pixel;
            }
        }
        
        __global__ void detect_text_regions(
            const unsigned char* input,
            float* text_scores,
            int width,
            int height,
            float threshold
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (idx < width && idy < height) {
                int pixel_idx = idy * width + idx;
                
                // Simple edge detection for text region identification
                if (idx > 0 && idx < width-1 && idy > 0 && idy < height-1) {
                    float gx = -input[pixel_idx - width - 1] + input[pixel_idx - width + 1] 
                              -2 * input[pixel_idx - 1] + 2 * input[pixel_idx + 1]
                              - input[pixel_idx + width - 1] + input[pixel_idx + width + 1];
                    
                    float gy = -input[pixel_idx - width - 1] - 2 * input[pixel_idx - width] - input[pixel_idx - width + 1]
                              + input[pixel_idx + width - 1] + 2 * input[pixel_idx + width] + input[pixel_idx + width + 1];
                    
                    float gradient = sqrtf(gx*gx + gy*gy);
                    text_scores[pixel_idx] = gradient > threshold ? 1.0f : 0.0f;
                } else {
                    text_scores[pixel_idx] = 0.0f;
                }
            }
        }
        
        } // extern "C"
        """
        return load_cuda_kernel(kernel_source, "ocr_preprocessing")
    
    def process_pdf_with_gpu_ocr(
        self,
        pdf_path: Path,
        *,
        output_path: Path | None = None,
        page_range: Tuple[int, int] | None = None,
        ocr_options: Dict[str, Any] | None = None,
        save_ocr_layer: bool = True
    ) -> Dict[int, str]:
        """Process PDF with GPU-accelerated OCR and optional persistent OCR layer."""
        
        options = ocr_options or {}
        contrast = options.get("contrast", 1.2)
        brightness = options.get("brightness", 10.0)
        gamma = options.get("gamma", 0.9)
        text_threshold = options.get("text_threshold", 50.0)
        
        results = {}
        pdf = pdfium.PdfDocument(pdf_path)
        
        try:
            total_pages = len(pdf)
            start_page, end_page = page_range or (0, total_pages)
            
            for page_num in range(start_page, min(end_page, total_pages)):
                page = pdf[page_num]
                
                # Render page to image
                bitmap = page.render(
                    scale=2.0,  # Higher resolution for better OCR
                    rotation=0,
                )
                pil_image = bitmap.to_pil()
                
                # Convert to grayscale numpy array
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')
                
                img_array = np.array(pil_image)
                height, width = img_array.shape
                
                # GPU preprocessing
                processed_img = self._gpu_preprocess_image(
                    img_array, contrast, brightness, gamma, text_threshold
                )
                
                # Convert back to PIL for OCR
                processed_pil = Image.fromarray(processed_img.astype(np.uint8), mode='L')
                
                # Perform OCR using Ollama
                extracted_text = self._perform_ocr_ollama(processed_pil, page_num)
                
                if extracted_text.strip():
                    results[page_num + 1] = extracted_text
                    
                    # Save OCR layer if requested
                    if save_ocr_layer and output_path:
                        self._save_ocr_layer(pdf_path, page_num, extracted_text, output_path)
                
        finally:
            pdf.close()
            
        return results
    
    def _gpu_preprocess_image(
        self,
        img_array: np.ndarray,
        contrast: float,
        brightness: float,
        gamma: float,
        text_threshold: float
    ) -> np.ndarray:
        """Apply GPU-accelerated image preprocessing for OCR."""
        
        height, width = img_array.shape
        
        # Allocate GPU memory
        with self.gpu_context:
            import cupy as cp
            
            # Upload image to GPU
            d_input = cp.asarray(img_array.astype(np.uint8))
            d_output = cp.empty_like(d_input)
            d_text_scores = cp.empty((height, width), dtype=cp.float32)
            
            # Configure kernel launch parameters
            threads_per_block = (16, 16)
            blocks_per_grid = (
                (width + threads_per_block[0] - 1) // threads_per_block[0],
                (height + threads_per_block[1] - 1) // threads_per_block[1]
            )
            
            # Apply preprocessing kernel
            self.ocr_kernel.preprocess_image_for_ocr(
                blocks_per_grid,
                threads_per_block,
                (d_input.data.ptr, d_output.data.ptr, width, height, 
                 contrast, brightness, gamma)
            )
            
            # Apply text detection kernel
            self.ocr_kernel.detect_text_regions(
                blocks_per_grid,
                threads_per_block,
                (d_input.data.ptr, d_text_scores.data.ptr, width, height, text_threshold)
            )
            
            # Download results
            processed_img = cp.asnumpy(d_output)
            text_scores = cp.asnumpy(d_text_scores)
            
            # Enhance text regions
            mask = text_scores > 0.5
            processed_img[mask] = np.clip(processed_img[mask] * 1.2, 0, 255)
            
            return processed_img
    
    def _perform_ocr_ollama(self, pil_image: Image.Image, page_num: int) -> str:
        """Perform OCR using Ollama model."""
        
        # Convert image to base64 for Ollama
        import base64
        from io import BytesIO
        
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Create OCR prompt
        ocr_prompt = f"""You are an OCR expert. Extract all text from this document page {page_num + 1}.

Requirements:
- Extract all visible text accurately
- Preserve line breaks and paragraph structure
- Include headers, footers, and marginal notes
- Use unicode characters for special symbols
- If text is unclear, indicate with [unclear] markers
- Return only the extracted text, no additional commentary"""

        # Call Ollama for OCR
        from knowledge3d.ingestion.ollama_manager import OllamaManager
        ollama = OllamaManager()
        
        result = ollama.chat(
            model=self.ocr_model,
            messages=[
                {"role": "system", "content": ocr_prompt},
                {"role": "user", "content": f"Page {page_num + 1} image: data:image/png;base64,{img_base64}"}
            ],
            temperature=0.1,
            options={"num_predict": 4096}
        )
        
        if result.returncode == 0:
            return result.output.strip()
        else:
            return f"[OCR failed: {result.stderr}]"
    
    def _save_ocr_layer(
        self,
        original_pdf: Path,
        page_num: int,
        ocr_text: str,
        output_path: Path
    ) -> None:
        """Save OCR text as persistent layer in PDF."""
        
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import TextStringObject, DictionaryObject, NameObject
        
        # Read original PDF
        reader = PdfReader(original_pdf)
        writer = PdfWriter()
        
        # Copy all pages
        for i in range(len(reader.pages)):
            writer.add_page(reader.pages[i])
        
        # Add OCR text as invisible layer to the specific page
        page = writer.pages[page_num]
        
        # Create text annotation with OCR content
        ocr_annotation = DictionaryObject()
        ocr_annotation.update({
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Text"),
            NameObject("/Contents"): TextStringObject(ocr_text),
            NameObject("/Rect"): [0, 0, 0, 0],  # Invisible
            NameObject("/C"): [1, 1, 1],  # White color (invisible)
            NameObject("/F"): 4,  # Hidden flag
            NameObject("/Name"): TextStringObject(f"OCR_Page_{page_num + 1}"),
            NameObject("/T"): TextStringObject("GPU_OCR_Processor"),
        })
        
        # Add annotation to page
        if "/Annots" not in page:
            page[NameObject("/Annots")] = []
        page["/Annots"].append(ocr_annotation)
        
        # Save with OCR layer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            writer.write(f)
    
    def check_gpu_memory_requirements(self, image_dims: Tuple[int, int]) -> bool:
        """Check if GPU has enough memory for processing."""
        
        width, height = image_dims
        # Estimate memory: input + output + text_scores + overhead
        estimated_bytes = width * height * 4 * 3 + 1024 * 1024 * 100  # 100MB overhead
        
        gpu_memory = self.gpu_context.get_available_memory()
        return gpu_memory > estimated_bytes


class OllamaSecondPassEnricher:
    """Ollama-assisted second pass enrichment for deeper semantic understanding."""
    
    def __init__(self, *, enrichment_model: str = "kimi-k2-thinking:cloud"):
        self.enrichment_model = enrichment_model
        
    def enrich_with_ollama(
        self,
        original_text: str,
        first_pass_results: Dict[str, Any],
        context_chunks: List[str],
        *,
        enrichment_options: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Perform Ollama-assisted second pass enrichment."""
        
        options = enrichment_options or {}
        depth_level = options.get("depth_level", "comprehensive")
        
        # Build enrichment prompt
        enrichment_prompt = f"""You are a knowledge enrichment expert. Perform a deep semantic analysis and enrichment of this content.

Original Text:
{original_text[:4000]}

First Pass Results:
{json.dumps(first_pass_results, indent=2)[:2000]}

Context Chunks:
{chr(10).join(context_chunks)[:2000]}

Enrichment Requirements (Level: {depth_level}):
1. Deepen semantic understanding beyond surface extraction
2. Identify implicit relationships and connections
3. Add missing conceptual layers (Form -> Meaning -> Rules -> Meta-Rules)
4. Enhance taxonomy, reality, and symbol references
5. Correct any factual inaccuracies from first pass
6. Add temporal, causal, and comparative relationships

Return enriched knowledge following the exact JSON schema provided."""

        # Call Ollama for enrichment
        from knowledge3d.ingestion.ollama_manager import OllamaManager
        ollama = OllamaManager()
        
        result = ollama.chat(
            model=self.enrichment_model,
            messages=[
                {"role": "system", "content": enrichment_prompt},
                {"role": "user", "content": "Perform deep semantic enrichment of this content"}
            ],
            temperature=0.2,
            options={"num_predict": 8192}
        )
        
        if result.returncode == 0:
            try:
                enriched_data = json.loads(result.output.strip())
                return enriched_data
            except json.JSONDecodeError:
                return {"error": "Invalid JSON from enrichment", "raw_output": result.output}
        else:
            return {"error": f"Enrichment failed: {result.stderr}"}