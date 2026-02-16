# Claude → Codex: PDF Ingestion Null Byte Fix

**Date:** February 14, 2026
**Issue:** Overnight PDF ingestion failed with "ValueError: embedded null byte"
**Root Cause:** PDF text extraction includes null bytes, which break subprocess.run() when passed to Ollama
**Priority:** CRITICAL FIX (blocking overnight ingestion)

---

## 🔴 **Error Analysis:**

**What happened:**
```
ValueError: embedded null byte

Location: subprocess.py:1796 in _execute_child
Chain: ollama_manager.py:61 query() → subprocess.run(["ollama", "run", model, prompt])
```

**Root cause:**
- PDFs contain binary data, null bytes, or corrupted text
- When extracting page text, null bytes get into classification prompts
- subprocess.run() rejects arguments containing null bytes
- Error happens on first PDF with problematic text

---

## 🛠️ **Fix: Sanitize Prompts Before Subprocess**

### **File 1: `knowledge3d/ingestion/ollama_manager.py`**

**Current code (line 52-67):**
```python
def query(
    self,
    model: str,
    prompt: str,
    timeout: float | None = None,
) -> OllamaQueryResult:
    """Run a non-conversational model call."""
    run_timeout = timeout if timeout is not None else self.default_timeout
    try:
        proc = subprocess.run(
            ["ollama", "run", model, prompt],  # ← PROBLEM: prompt may contain null bytes
            check=False,
            capture_output=True,
            text=True,
            timeout=run_timeout,
        )
```

**Fixed code:**
```python
def query(
    self,
    model: str,
    prompt: str,
    timeout: float | None = None,
) -> OllamaQueryResult:
    """Run a non-conversational model call."""
    run_timeout = timeout if timeout is not None else self.default_timeout

    # Sanitize prompt: remove null bytes and other problematic characters
    sanitized_prompt = self._sanitize_prompt(prompt)

    try:
        proc = subprocess.run(
            ["ollama", "run", model, sanitized_prompt],  # ← FIXED: use sanitized prompt
            check=False,
            capture_output=True,
            text=True,
            timeout=run_timeout,
        )
        return OllamaQueryResult(
            model=model,
            output=(proc.stdout or "").strip(),
            returncode=proc.returncode,
            stderr=(proc.stderr or "").strip(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return OllamaQueryResult(
            model=model,
            output="",
            returncode=1,
            stderr=str(exc),
        )

def _sanitize_prompt(self, prompt: str) -> str:
    """
    Sanitize prompt for subprocess safety.

    Removes:
    - Null bytes (\x00)
    - Other control characters that break subprocess
    - Excessive whitespace

    Returns: Clean prompt safe for subprocess.run()
    """
    # Remove null bytes (primary cause of "embedded null byte" error)
    clean = prompt.replace('\x00', '')

    # Remove other problematic control characters (keep newlines/tabs)
    clean = ''.join(c if c >= ' ' or c in '\n\t' else ' ' for c in clean)

    # Normalize whitespace (collapse multiple spaces)
    clean = ' '.join(clean.split())

    # Ensure valid UTF-8 encoding
    clean = clean.encode('utf-8', errors='ignore').decode('utf-8')

    # Truncate if too long (Ollama has limits)
    MAX_PROMPT_LENGTH = 100000  # ~100K chars
    if len(clean) > MAX_PROMPT_LENGTH:
        clean = clean[:MAX_PROMPT_LENGTH] + "\n[... truncated for length ...]"

    return clean
```

---

### **File 2: `knowledge3d/ingestion/pdf_classifier.py`** (Additional Safety)

**Add validation before calling Ollama:**

```python
def classify_page(
    self,
    pdf_path: Path,
    page_num: int,
    page_text: str,
    context_pages: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Classify single page with LLM."""

    # Sanitize page text before using in prompts
    page_text = self._sanitize_text(page_text)

    # Check cache first
    cache = self._load_cache(pdf_path)
    cached = self._check_cache(cache, page_num)
    if cached:
        return cached

    # ... rest of method

def _sanitize_text(self, text: str) -> str:
    """
    Sanitize extracted PDF text.

    Removes null bytes and other problematic characters
    that could break Ollama subprocess calls.
    """
    # Remove null bytes
    clean = text.replace('\x00', '')

    # Remove other control characters (keep newlines)
    clean = ''.join(c if c >= ' ' or c == '\n' else ' ' for c in clean)

    # Normalize excessive whitespace
    clean = '\n'.join(line.strip() for line in clean.split('\n'))

    # Ensure UTF-8
    clean = clean.encode('utf-8', errors='ignore').decode('utf-8')

    # Truncate very long pages
    MAX_PAGE_LENGTH = 50000
    if len(clean) > MAX_PAGE_LENGTH:
        clean = clean[:MAX_PAGE_LENGTH] + "\n[... page truncated ...]"

    return clean
```

---

## 🧪 **Testing:**

### **Unit Test: Null Byte Handling**

Create `tests/test_ollama_manager_sanitization.py`:

```python
"""Test Ollama manager prompt sanitization."""

import pytest
from knowledge3d.ingestion.ollama_manager import OllamaModelManager


def test_sanitize_prompt_removes_null_bytes():
    """Verify null bytes are removed from prompts."""
    manager = OllamaModelManager()

    # Prompt with null bytes (simulates corrupted PDF text)
    dirty_prompt = "Hello\x00World\x00Test"

    # Should remove null bytes
    clean = manager._sanitize_prompt(dirty_prompt)
    assert '\x00' not in clean
    assert clean == "Hello World Test"


def test_sanitize_prompt_handles_control_characters():
    """Verify control characters are cleaned."""
    manager = OllamaModelManager()

    # Prompt with various control chars
    dirty_prompt = "Hello\x01\x02\x03World\x0BTest"

    clean = manager._sanitize_prompt(dirty_prompt)
    assert clean == "Hello World Test"


def test_sanitize_prompt_preserves_newlines():
    """Verify newlines and tabs are preserved."""
    manager = OllamaModelManager()

    dirty_prompt = "Line 1\nLine 2\tTabbed"

    clean = manager._sanitize_prompt(dirty_prompt)
    assert '\n' in clean
    assert '\t' in clean


def test_sanitize_prompt_truncates_long_text():
    """Verify very long prompts are truncated."""
    manager = OllamaModelManager()

    # Extremely long prompt
    long_prompt = "A" * 200000

    clean = manager._sanitize_prompt(long_prompt)
    assert len(clean) < 150000  # Should be truncated
    assert "truncated" in clean.lower()


def test_sanitize_prompt_handles_utf8_errors():
    """Verify invalid UTF-8 is handled gracefully."""
    manager = OllamaModelManager()

    # Invalid UTF-8 sequence
    dirty_prompt = "Hello " + chr(0xD800) + " World"  # Invalid surrogate

    # Should not raise, should return clean text
    clean = manager._sanitize_prompt(dirty_prompt)
    assert isinstance(clean, str)
```

---

## 🚀 **Implementation Steps:**

### **Step 1: Apply Fix**

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

# Edit ollama_manager.py
# Add _sanitize_prompt() method
# Update query() to use sanitized prompt

# Edit pdf_classifier.py
# Add _sanitize_text() method
# Call before using page text in prompts
```

### **Step 2: Test Sanitization**

```bash
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q tests/test_ollama_manager_sanitization.py
```

### **Step 3: Restart Overnight Ingestion**

```bash
# Kill current tmux session (already failed)
tmux kill-session -t k3d_pdf_ingestion

# Start fresh with fixed code
tmux new -s k3d_pdf_ingestion
bash scripts/run_overnight_pdf_ingestion.sh

# Detach: Ctrl+b then d
```

---

## 📊 **Expected Behavior After Fix:**

**Before (current):**
```
ValueError: embedded null byte
→ Ingestion stops on first PDF with null bytes
→ No output generated
```

**After (fixed):**
```
Null bytes sanitized silently
→ Classification proceeds with clean text
→ All 1,952 PDFs processed successfully
→ ~15,000-25,000 entries generated
```

---

## 🔍 **Additional Robustness:**

### **Optional: PDF Text Extraction Hardening**

If issues persist after sanitization, add fallback PDF extraction:

```python
def _extract_pdf_text_safe(pdf_path: Path, page_num: int) -> str:
    """
    Extract PDF page text with error handling.

    Falls back to OCR if extraction fails or produces garbage.
    """
    try:
        # Primary: pypdf
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            page = reader.pages[page_num]
            text = page.extract_text()

        # Validate extracted text
        if _is_garbage_text(text):
            # Fallback: OCR (if needed)
            text = _extract_with_ocr(pdf_path, page_num)

        return text

    except Exception as e:
        # Log error, return empty (classify as non-knowledge)
        print(f"PDF extraction failed for {pdf_path}:{page_num}: {e}")
        return ""


def _is_garbage_text(text: str) -> bool:
    """Detect if extracted text is garbage (too many control chars)."""
    if not text:
        return True

    # Count control characters (excluding newlines/tabs)
    control_chars = sum(1 for c in text if c < ' ' and c not in '\n\t')

    # If >10% control chars, likely garbage
    return (control_chars / len(text)) > 0.10
```

---

## 💡 **Root Cause Summary:**

1. **PDF files contain binary/corrupted data** (common with scanned docs)
2. **PyPDF2 extraction includes null bytes** from binary sections
3. **subprocess.run() rejects null bytes** in arguments (Python security)
4. **Solution:** Sanitize all text before subprocess calls

---

## 🎯 **Success Criteria (After Fix):**

- ✅ Overnight ingestion runs to completion (no ValueError)
- ✅ All 1,952 PDFs processed (or skipped with clear errors)
- ✅ Output file generated: `full_pdf_payloads_overnight_*.jsonl`
- ✅ Entry count: 15,000-25,000 (as expected)
- ✅ Cache populated: `../Knowledge3D.local/pdf_cache/*.json`

---

**Priority:** CRITICAL - Fix before re-running overnight ingestion
**ETA:** 1-2 hours (implementation + testing + restart)
**Impact:** Unblocks full knowledge construction pipeline

---

**Handoff prepared by:** Claude (Debugging)
**Date:** February 14, 2026
**Status:** Fix ready for Codex implementation
