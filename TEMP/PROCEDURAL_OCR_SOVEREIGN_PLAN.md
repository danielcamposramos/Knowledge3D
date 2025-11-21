# Procedural OCR — Sovereign Plan (Letters → House/Galaxy)

**Goal**: Replace pixel-bound OCR with a procedural pipeline: PDF → procedural glyph programs → Galaxy match → logical resolver. Weights stay logical; data lives in House/Galaxy.

## Pipeline Outline
1) **PDF ingestion (procedural first)**  
   - If fonts embedded: resolve font family → reuse already ingested glyph stars (no re-harvest). Map codepoints to stored RPN glyphs directly and compose words from these glyph stars.  
   - If only images: patch-wise proceduralization (ternary texture/edge-to-stroke approximator) to produce coarse stroke programs; no pixel tensors remain after this step.
2) **Procedural normalization + augmentation**  
   - Use harvested multi-font RPN glyphs as base (31 scripts, all fonts under system + /K3D/Knowledge3D.local/fonts).  
   - Procedural augmentations: stroke width/slant, jitter, blur, print/scan artifacts, partial erosion/dilation, ligature splits/merges; all applied in RPN/geometry space, not pixels.
3) **Embedding + Galaxy lookup**  
   - Execute RPN glyphs (scan-derived and harvested) via GPU drawing executor → embeddings.  
   - Nearest-neighbor in Galaxy over procedural glyph embeddings; constrain by script/language priors and layout context. If fonts were embedded/matched, bypass lookup and use the exact glyph star.
4) **Logical resolver (weights = logic)**  
   - Use specialist weights for invariance and disambiguation (script→char gating, language bigrams/pairs like “rr”, “ch” per language, position/spacing hints).  
   - When confidence low, attach multiple candidates with scores; store back to House.
5) **Write-back**  
   - Store recognized text as **composed stars**: words become composition of character stars (symbolic link style), enriched with grammar/language metadata. Keep links to glyph stars (when matched) and procedural evidence (when approximated); no pixels stored.

## Data + Coverage
- Fonts: already proceduralized across 31 script files (`fonts_*_procedural.jsonl`), covering system + `/K3D/Knowledge3D.local/fonts/` (external CJK, etc.).  
- Audio: procedural audio stars ready (en/es/pt-br/ja/zh). Not critical for OCR but available for cross-modal checks.  
- Procedural scan-to-stroke codec: needed as a fallback when contours are absent; design as a GPU edge/segment approximator that emits coarse RPN strokes.

## Training/Testing
- Training data: use procedural glyphs + procedural augmentations (no pixel tensors). Add a small set of proceduralized scans (PDF pages → procedural strokes) for domain adaptation.  
- Validation: compare against pixel OCR baselines on held-out scanned PDFs; measure CER/WER and robustness to noise/blur/low-res.  
- Integration: swap pixel input for procedural embeddings in OCR specialist; use Galaxy lookup + resolver.

## Risks & Mitigations
- **Contour loss in scanned images**: stroke approximator may be coarse. Mitigate by heavy procedural augmentation during training and Galaxy nearest-neighbor rescue with script priors.  
- **Layout/segmentation**: still need reliable line/word segmentation. Keep minimal layout detection (bounding boxes) but run recognizer on proceduralized glyph clusters instead of crops.  
- **Performance**: proceduralization cost for large PDFs. Use batching and GPU kernels; avoid CPU NumPy paths.

## Will it work vs pixel OCR (incl. DeepSeek-OCR)?
- Likely stronger on domain shift/noise once trained, because invariances are baked in procedurally and Galaxy lookup can rescue low-quality strokes.  
- Early-stage accuracy will trail mature pixel OCR until the procedural scan-to-stroke codec and training are tuned. Expect improvement as the procedural augmentations and Galaxy matching mature.  
- Advantages over pixel OCR: sovereignty (GPU-native, no raster tensors kept), explainability (procedural strokes), and cross-modal fusion.  
- Risks: if stroke approximation is too coarse, CER/WER will lag strong pixel models; must validate against a pixel baseline on real scans.

## Next steps (engineering)
1) Add `scripts/proceduralize_pdf.py`: font-outline → RPN; image fallback → stroke approximator (ternary texture + edge to RPN).  
2) Add procedural augmentor in RPN space for OCR training.  
3) Wire OCR specialist to consume procedural glyph embeddings + Galaxy lookup (replace pixel frontend).  
4) Benchmark against pixel OCR on scanned PDFs; iterate stroke approximator as needed.
