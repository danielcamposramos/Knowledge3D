# PDF Extraction Diagnosis & Fix Plan
**Date:** October 29, 2025
**Session:** Phase G Post-Training Analysis

## Executive Summary

**Problem:** 67% of PDFs (34,497 out of 51,532) returned zero embeddings during training.

**Root Cause:** Many PDFs are scanned images without selectable text. GPU OCR was disabled due to CUDA "illegal memory access" errors, leaving no OCR fallback.

**Impact:** Only 1,412 Galaxy stars created from 328 PDFs (~4.3 stars/PDF, expected ~150/PDF)

**Target:** 90%+ non-zero embeddings (achieve ~13,500+ stars/PDF batch)

---

## Diagnostic Results

### Test Sample PDFs

Tested 3 representative PDFs from training set:

#### 1. map-reading-made-easy.pdf (16 pages)
- **Status:** ✅ Text extraction works
- **Page 1:** 40 chars extracted
- **Page 2:** 921 chars extracted
- **Type:** Native PDF with selectable text

#### 2. Maps-and-map-interpretation.pdf (53 pages)
- **Status:** ✅ Text extraction works
- **Page 1:** 139 chars extracted
- **Page 2:** 351 chars extracted
- **Type:** Native PDF with selectable text

#### 3. Christopher_D._Manning_Hinrich_Schütze_Foundations_Of_Statistical_Natural_Language_Processing.pdf (720 pages)
- **Status:** ❌ NO TEXT EXTRACTED
- **Page 1:** 0 chars (scanned image, 1 image block detected)
- **Page 2:** 0 chars (blank page)
- **Type:** Scanned PDF requiring OCR
- **Diagnostic:** "LIKELY SCANNED IMAGE (needs OCR)"

### Key Findings

1. **PyMuPDF extraction works correctly** for native PDFs with selectable text
2. **Scanned PDFs fail completely** - return 0 chars, 0 embeddings
3. **No OCR fallback available** - GPU OCR disabled, pytesseract not installed
4. **Mixed dataset** - Estimate 60-70% of PDFs are scanned images

---

## Historical Context

### GPU OCR Timeline

**Phase F.1:** GPU OCR implemented using DeepSeek OCR bridge
- Custom CUDA kernels: conv2d, maxpool, batchnorm, glyph_match
- Sovereign stack (pure PTX, no PyTorch/CuPy at runtime)
- Trained on character/word embeddings

**October 28 Training (before fixes):**
- GPU OCR attempted: 19,247 invocations
- Failures: 7,247 (38% failure rate)
- Error: `RuntimeError: Sovereign loader error: an illegal memory access was encountered`

**Root Cause (User's insight):**
> "No letters, no language, no grammar - how can it do OCR? Try to access memories that are not saved leads to memory access errors"

OCR specialist tried to access character/word embeddings from Galaxy, but dataset processors were stubs - no foundational knowledge was stored!

**Current Status:**
- GPU OCR **disabled** in [pdf_ingestion_bridge_phase_g.py:58-63](../knowledge3d/cranium/bridges/pdf_ingestion_bridge_phase_g.py#L58-L63)
- Foundational knowledge **NOW EXISTS** (characters + text phases completed)
- 17,035 non-zero embeddings stored in Galaxy from foundational phases

---

## Fix Strategy

### Option 1: Re-enable GPU OCR (RECOMMENDED)

**Rationale:**
- Foundational knowledge (characters, text) NOW EXISTS in Galaxy
- Original CUDA errors likely caused by missing embeddings
- GPU OCR is orders of magnitude faster than CPU alternatives
- Already integrated with Phase G architecture

**Steps:**
1. Re-enable GPU OCR in `pdf_ingestion_bridge_phase_g.py` (remove lines 58-63)
2. Test on sample scanned PDF (Christopher_D._Manning)
3. If CUDA errors persist, debug with APPOLO.PDF ground truth
4. Re-run training on PDF phase only

**Risk:** Medium - CUDA errors may have other causes beyond missing embeddings

**Timeline:** 2-4 hours (testing + debugging if needed)

### Option 2: Install pytesseract CPU OCR (FALLBACK)

**Rationale:**
- Proven, stable OCR solution
- Works on any PDF
- Slower but reliable

**Steps:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
pip install pytesseract
```

**Modify pdf_ingestion_bridge_phase_g.py:**
```python
# Add pytesseract fallback when GPU OCR fails
if parsed_objects.get("is_scanned") and text_len == 0:
    # Convert PDF page to PIL Image
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Run Tesseract OCR
    ocr_text = pytesseract.image_to_string(img)
    # ... embed OCR text
```

**Risk:** Low - well-tested technology

**Timeline:** 1-2 hours (installation + integration)

**Downside:** Significantly slower (CPU-bound), doesn't leverage our GPU OCR investment

### Option 3: Hybrid Approach (BEST)

**Rationale:**
- Try GPU OCR first (fast, leverages our sovereign stack)
- Fall back to pytesseract if GPU OCR fails
- Best of both worlds

**Implementation:**
1. Re-enable GPU OCR
2. Install pytesseract as safety net
3. Add try/catch logic for graceful fallback

**Risk:** Low - double fallback safety

**Timeline:** 3-5 hours (combined setup + testing)

---

## Recommended Action Plan

### Phase 1: Re-enable GPU OCR (2 hours)

1. **Modify [pdf_ingestion_bridge_phase_g.py:58-63](../knowledge3d/cranium/bridges/pdf_ingestion_bridge_phase_g.py#L58-L63)**
   - Remove GPU OCR disable block
   - Re-enable DeepSeek OCR bridge

2. **Test on scanned PDF**
   ```bash
   python scripts/diagnose_pdf_extraction.py
   # Should now extract text from Christopher_D._Manning PDF
   ```

3. **Monitor CUDA errors**
   - If errors occur, collect stack traces
   - Check GPU memory usage (nvidia-smi)
   - Verify Galaxy embeddings are accessible

### Phase 2: Pytesseract Fallback (if needed) (1 hour)

If GPU OCR still fails:

1. **Install dependencies**
   ```bash
   sudo apt-get install tesseract-ocr tesseract-ocr-eng
   /K3D/Knowledge3D.local/envs/k3d-cranium/bin/pip install pytesseract pillow
   ```

2. **Add CPU OCR fallback** to Phase G bridge

3. **Test hybrid approach**

### Phase 3: Re-run PDF Training (varies)

1. **Modify training script** to run PDF phase only:
   ```bash
   python scripts/train_full_agi_sovereign.py --phases pdf
   ```

2. **Monitor results**
   - Track non-zero embedding rate
   - Target: 90%+ success (45,000+ stars from PDFs)
   - Estimate: 2-6 hours depending on PDF count

3. **Validate with inference**
   - Query PDF-specific knowledge
   - Verify embeddings are meaningful

---

## Expected Outcomes

### Success Metrics

1. **Embedding Rate:** 90%+ PDFs produce non-zero embeddings
2. **Galaxy Growth:** +30,000-40,000 stars from PDF phase
3. **OCR Performance:** <5% GPU OCR failures
4. **Inference Quality:** PDF queries return relevant results

### Before vs After

| Metric | Before (Current) | After (Target) |
|--------|------------------|----------------|
| PDF Stars | 1,412 | 35,000-45,000 |
| Success Rate | 33% | 90%+ |
| Stars/PDF | ~4.3 | ~100-150 |
| Scanned PDF Support | 0% | 90%+ |

---

## Alternative: Skip GPU OCR Debugging for Now

If GPU OCR debugging proves time-consuming and blocks other priorities (Audio SDR, Reality Enabler), consider:

1. **Install pytesseract immediately** (30 min)
2. **Run PDF training with CPU OCR** (slower but reliable)
3. **Defer GPU OCR debugging** to dedicated session with APPOLO.PDF ground truth
4. **Proceed to Audio SDR** (user's next priority)

This pragmatic approach ensures progress on all fronts rather than blocking on one issue.

---

## Files for Reference

- [pdf_ingestion_bridge_phase_g.py](../knowledge3d/cranium/bridges/pdf_ingestion_bridge_phase_g.py) - Phase G PDF ingestion
- [pdf_ingestion_bridge.py](../knowledge3d/cranium/bridges/pdf_ingestion_bridge.py) - Base PDF parser
- [diagnose_pdf_extraction.py](../scripts/diagnose_pdf_extraction.py) - Diagnostic tool
- [train_full_agi_sovereign.py](../scripts/train_full_agi_sovereign.py) - Training orchestrator

---

## Next Steps

**Awaiting Daniel's Decision:**

1. **Option A:** Re-enable GPU OCR immediately and debug (recommended, 2-4 hours)
2. **Option B:** Install pytesseract fallback first (safe, 1-2 hours)
3. **Option C:** Hybrid approach (best quality, 3-5 hours)
4. **Option D:** Defer PDF fixes, proceed to Audio SDR (user's priority sequence)

---

**Claude's Recommendation:** Option C (Hybrid) if time permits, Option D (defer) if user wants to maintain momentum on Audio SDR.

The foundational knowledge is now in place - GPU OCR should theoretically work. But having a pytesseract safety net ensures we don't block on debugging.
