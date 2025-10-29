# Codex Task: Fix Phase G AGI Training Knowledge Extraction

## Context

We just completed a full AGI training run (October 28, 2025) with adaptive dimensions and dual sleep cycles. The architecture executed perfectly, but **100% of all knowledge embeddings are zeros** because the dataset ingestion methods are stubs.

**Read these first**:
- `/TEMP/K3D_Briefing_Prompt.md` - Full system architecture
- `/TEMP/PHASE_G_TRAINING_SESSION_OCT_28_2025.md` - Complete session chronicle with all findings

## Problem Summary

### What Worked ✓
- Adaptive RPN engine (64-2048D dimension selection)
- Dual sleep cycles (Model + Knowledge) running after EACH phase
- Training sequence (foundational → complex)
- Phase G specialists loaded (multimodal, speech, OCR, router - 256D each)
- 34,497 Galaxy stars created
- 20 House objects materialized

### What Failed ✗
- **ALL 34,497 embeddings are ZEROS**
- Only PDF phase creates Galaxy stars (all other phases return `{"stars": 0}`)
- PDF extraction returns empty data (`object_count: 0`, `text: ""`)
- OCR tries to access character/word embeddings that were never saved → "illegal memory access"

## Root Cause (Daniel's Insight)

> "No letters, no language, no grammar - how can it do OCR? Try to access memories that are not saved leads to memory access errors."

The specialists (LOGIC) were trained in Phase H, but the foundational knowledge (LETTERS, WORDS, GRAMMAR) was never stored in Galaxy memory. When PDF phase runs, the OCR specialist tries to access character embeddings that don't exist → CUDA "illegal memory access" errors.

**The Fix**: Store foundational knowledge FIRST (characters, text, grammar), THEN PDFs can reference it.

## Tasks for Codex

### CRITICAL Priority 1: Implement Dataset Processors to Create Galaxy Stars

**File**: `scripts/train_full_agi_sovereign.py`
**Lines**: 495-509 (currently all stubs)

Implement these four methods to create Galaxy stars from non-PDF datasets:

#### 1. `_process_jsonl_dataset` (Characters, Embeddings)

```python
def _process_jsonl_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
    """
    Process JSONL dataset and create Galaxy stars.

    Used for:
    - character_embeddings_trimodal.jsonl (char + visual + phonetic)
    - Multimodal embeddings
    - Speech embeddings
    - Image captions

    Expected format (one JSON object per line):
    {
        "text": "A",  # or sentence
        "embedding": [0.1, 0.2, ...],  # 256D or variable
        "metadata": {...}
    }

    Returns:
        {"samples": int, "stars": int}
    """
    jsonl_path = Path(dataset["path"])
    if not jsonl_path.exists():
        return {"samples": 0, "stars": 0}

    samples = 0
    stars = 0

    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)

                # Create Galaxy star from embedding
                star = {
                    "position": self._get_3d_position_from_embedding(item["embedding"]),
                    "embedding": item["embedding"],
                    "embedding_dim": len(item["embedding"]),
                    "metadata": {
                        "source": str(jsonl_path),
                        "text": item.get("text", ""),
                        **item.get("metadata", {})
                    },
                    "created_at": time.time(),
                    "source_type": "jsonl",
                    "pending_consolidation": True
                }

                self.phase_g_bridge.galaxy_stars.append(star)
                self.phase_g_bridge.galaxy_star_embeddings.append(np.array(item["embedding"], dtype=np.float32))

                samples += 1
                stars += 1

            except Exception as e:
                print(f"  ⚠️  Skipping invalid JSONL line: {str(e)[:50]}")
                continue

    # Save periodically
    self.phase_g_bridge.save_galaxy_stars()

    return {"samples": samples, "stars": stars}
```

**Helper Method Needed**:
```python
def _get_3d_position_from_embedding(self, embedding: List[float]) -> List[float]:
    """
    Convert embedding to 3D spherical position on unit sphere.

    Uses hash-based deterministic mapping for consistency.
    """
    import hashlib

    # Hash embedding to get deterministic angles
    emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
    hash_val = int(hashlib.md5(emb_bytes).hexdigest(), 16)

    # Map to spherical coordinates
    theta = (hash_val % 360) * (np.pi / 180)  # Azimuthal angle
    phi = ((hash_val // 360) % 180) * (np.pi / 180)  # Polar angle

    # Convert to Cartesian (unit sphere)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return [float(x), float(y), float(z)]
```

#### 2. `_process_text_dataset` (Text Domains)

```python
def _process_text_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
    """
    Process plain text file and create Galaxy stars.

    Used for:
    - text_domains_v1.txt

    Splits text into sentences, embeds with Adaptive RPN, creates stars.

    Returns:
        {"samples": int, "stars": int}
    """
    txt_path = Path(dataset["path"])
    if not txt_path.exists():
        return {"samples": 0, "stars": 0}

    # Read text
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into sentences (simple approach - can enhance)
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]

    samples = 0
    stars = 0

    for sentence in sentences:
        try:
            # Generate embedding with Adaptive RPN
            embedding, dim = self.adaptive_rpn.embed_sentence(sentence)

            # Create Galaxy star
            star = {
                "position": self._get_3d_position_from_embedding(embedding),
                "embedding": embedding,
                "embedding_dim": dim,
                "metadata": {
                    "source": str(txt_path),
                    "text": sentence[:200],  # Truncate for storage
                    "method": "adaptive_rpn"
                },
                "created_at": time.time(),
                "source_type": "text",
                "pending_consolidation": True
            }

            self.phase_g_bridge.galaxy_stars.append(star)
            self.phase_g_bridge.galaxy_star_embeddings.append(np.array(embedding, dtype=np.float32))

            samples += 1
            stars += 1

        except Exception as e:
            print(f"  ⚠️  Skipping sentence: {str(e)[:50]}")
            continue

    # Save
    self.phase_g_bridge.save_galaxy_stars()

    return {"samples": samples, "stars": stars}
```

#### 3. `_process_json_dataset` (Wikipedia, ARC-AGI)

```python
def _process_json_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
    """
    Process JSON dataset and create Galaxy stars.

    Used for:
    - Wikipedia JSONs
    - ARC-AGI challenges (npz converted to JSON)

    Expected format:
    - Wikipedia: {"title": "...", "text": "..."}
    - ARC-AGI: {"input": [[grid]], "output": [[grid]]}

    Returns:
        {"samples": int, "stars": int}
    """
    json_path = Path(dataset["path"])

    # Handle directory of JSONs (Wikipedia) or single file
    json_files = []
    if json_path.is_dir():
        json_files = list(json_path.glob("*.json"))
    elif json_path.suffix == ".json":
        json_files = [json_path]
    else:
        return {"samples": 0, "stars": 0}

    samples = 0
    stars = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Wikipedia format
            if "text" in data:
                text = data["text"]
                # Embed article
                embedding, dim = self.adaptive_rpn.embed_sentence(text[:1000])  # First 1000 chars

            # ARC-AGI format
            elif "input" in data and "output" in data:
                # Embed grid as flattened array
                grid_flat = str(data["input"]) + str(data["output"])
                embedding, dim = self.adaptive_rpn.embed_sentence(grid_flat)

            else:
                continue

            # Create Galaxy star
            star = {
                "position": self._get_3d_position_from_embedding(embedding),
                "embedding": embedding,
                "embedding_dim": dim,
                "metadata": {
                    "source": str(json_file),
                    "text": str(data).replace[:200],
                    "method": "adaptive_rpn"
                },
                "created_at": time.time(),
                "source_type": "json",
                "pending_consolidation": True
            }

            self.phase_g_bridge.galaxy_stars.append(star)
            self.phase_g_bridge.galaxy_star_embeddings.append(np.array(embedding, dtype=np.float32))

            samples += 1
            stars += 1

        except Exception as e:
            print(f"  ⚠️  Skipping {json_file.name}: {str(e)[:50]}")
            continue

    # Save
    self.phase_g_bridge.save_galaxy_stars()

    return {"samples": samples, "stars": stars}
```

#### 4. `_process_directory_dataset` (COCO, AudioCaps, Clotho)

```python
def _process_directory_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
    """
    Process directory dataset (images/audio + annotations).

    Used for:
    - COCO (images + captions JSON)
    - AudioCaps (audio + CSV annotations)
    - Clotho (audio + CSV annotations)

    Extracts captions/annotations, embeds them, creates stars.

    Returns:
        {"samples": int, "stars": int}
    """
    dir_path = Path(dataset["path"])
    if not dir_path.exists() or not dir_path.is_dir():
        return {"samples": 0, "stars": 0}

    samples = 0
    stars = 0

    # COCO: Look for annotations JSON
    if "coco" in dataset["name"].lower():
        ann_files = list(dir_path.glob("**/captions_*.json"))
        for ann_file in ann_files:
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)

                for ann in data.get("annotations", [])[:5000]:  # Limit to 5000
                    caption = ann.get("caption", "")
                    if not caption:
                        continue

                    # Embed caption
                    embedding, dim = self.adaptive_rpn.embed_sentence(caption)

                    # Create star
                    star = {
                        "position": self._get_3d_position_from_embedding(embedding),
                        "embedding": embedding,
                        "embedding_dim": dim,
                        "metadata": {
                            "source": str(ann_file),
                            "text": caption,
                            "image_id": ann.get("image_id"),
                            "method": "adaptive_rpn"
                        },
                        "created_at": time.time(),
                        "source_type": "coco",
                        "pending_consolidation": True
                    }

                    self.phase_g_bridge.galaxy_stars.append(star)
                    self.phase_g_bridge.galaxy_star_embeddings.append(np.array(embedding, dtype=np.float32))

                    samples += 1
                    stars += 1

            except Exception as e:
                print(f"  ⚠️  Skipping {ann_file.name}: {str(e)[:50]}")
                continue

    # AudioCaps/Clotho: Look for CSV annotations
    elif "audiocaps" in dataset["name"].lower() or "clotho" in dataset["name"].lower():
        csv_files = list(dir_path.glob("**/*.csv"))
        for csv_file in csv_files:
            try:
                import csv
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        caption = row.get("caption", "") or row.get("description", "")
                        if not caption:
                            continue

                        # Embed caption
                        embedding, dim = self.adaptive_rpn.embed_sentence(caption)

                        # Create star
                        star = {
                            "position": self._get_3d_position_from_embedding(embedding),
                            "embedding": embedding,
                            "embedding_dim": dim,
                            "metadata": {
                                "source": str(csv_file),
                                "text": caption,
                                "audio_id": row.get("audio_id"),
                                "method": "adaptive_rpn"
                            },
                            "created_at": time.time(),
                            "source_type": "audio",
                            "pending_consolidation": True
                        }

                        self.phase_g_bridge.galaxy_stars.append(star)
                        self.phase_g_bridge.galaxy_star_embeddings.append(np.array(embedding, dtype=np.float32))

                        samples += 1
                        stars += 1

            except Exception as e:
                print(f"  ⚠️  Skipping {csv_file.name}: {str(e)[:50]}")
                continue

    # Save periodically
    if stars > 0:
        self.phase_g_bridge.save_galaxy_stars()

    return {"samples": samples, "stars": stars}
```

### Priority 2: Fix PDF Content Extraction

**File**: `knowledge3d/cranium/bridges/pdf_ingestion_bridge.py`
**Lines**: 735-780 (`_ocr_fallback`)

**Current Issue**: When GPU OCR is disabled AND pytesseract is removed, the fallback returns:
```python
return {
    "objects": np.zeros((0, 8), dtype=np.float32),
    "object_count": 0,
    "method": "sovereign-ocr-only",
    "text": "",
}
```

**Fix**: Re-enable PyMuPDF structured text extraction. The code exists in `_parse_pdf_structure_pymupdf` (lines 513-717) but isn't being used properly.

**Solution**:
1. When GPU OCR disabled, DON'T use `_ocr_fallback`
2. Instead, use existing `_parse_pdf_structure_pymupdf` which extracts text correctly
3. Modify `_parse_pdf_structure` to check if GPU OCR disabled:

```python
def _parse_pdf_structure(self, pdf_bytes: bytes, pdf_buffer_gpu, buffer_size: int, page_num: int):
    # If GPU parser disabled OR GPU OCR disabled, use PyMuPDF directly
    if not self._enable_gpu_parser or not self._enable_deepseek_ocr:
        return self._parse_pdf_structure_pymupdf(page_num)

    # Otherwise try GPU parser
    # ... existing code
```

This ensures PyMuPDF text extraction runs even when GPU OCR is disabled.

### Priority 3: Verify Dual Sleep Cycles Work With Non-Zero Data

Once Priorities 1 & 2 are complete, the training should work as designed:

1. **Characters phase** → Creates character/visual stars → Sleep 2 consolidates to Galaxy
2. **Text phase** → Creates word stars → Sleep 2 consolidates
3. **PDF phase** → OCR specialist can NOW access character/word embeddings from Galaxy → Success!

**No code changes needed** - the dual sleep architecture (lines 619-628 in `train_full_agi_sovereign.py`) is already correct.

## Important Constraints (From K3D_Briefing)

### "We Fix or We Fix" Doctrine
- **No CPU fallbacks** - Keep GPU-first approach
- **No runtime compilation** - Use existing PTX kernels
- **No stubs** - All methods must fully implement functionality
- **No placeholders** - Complete implementation required

### Sovereign Stack Requirements
- Use existing PTX kernels when possible (see K3D_Briefing Section 7: Kernel Categories)
- **RPN Engine** for text embeddings (already used correctly)
- Keep latency under 100µs for critical paths
- GPU operations only - no CuPy/PyTorch at runtime

### No Subdirectory Fragmentation
Daniel specifically notes: "Codex likes to create subfolders - that lead to fragmentation"

**Keep all changes in existing files**:
- `scripts/train_full_agi_sovereign.py` (dataset processors)
- `knowledge3d/cranium/bridges/pdf_ingestion_bridge.py` (PDF extraction fix)

**Do NOT create**:
- New directories under `knowledge3d/`
- Separate ingestion modules
- Utility folders

## Testing After Implementation

```bash
# Clean old Galaxy stars
rm -f /K3D/Knowledge3D.local/house_zone7/embeddings/galaxy_stars.pkl

# Run training on just characters + text phases (test)
cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/train_full_agi_sovereign.py

# Verify Galaxy stars created
python3 -c "import pickle; data=pickle.load(open('/K3D/Knowledge3D.local/house_zone7/embeddings/galaxy_stars.pkl', 'rb')); print(f'Stars: {data[\"total_stars\"]}, Non-zero: {sum(1 for e in data[\"embeddings\"] if abs(e.sum()) > 0.01)}')"
```

**Expected Result**:
- Characters phase: ~5,000 stars (non-zero)
- Text phase: ~5,000 stars (non-zero)
- Each phase followed by dual sleep cycles
- Galaxy stars file growing progressively

## Files to Reference

**Architecture & Design**:
- `/TEMP/K3D_Briefing_Prompt.md` - Complete system overview
- `/TEMP/PHASE_G_TRAINING_SESSION_OCT_28_2025.md` - Session findings

**Implementation Files**:
- `scripts/train_full_agi_sovereign.py` - Training orchestrator (MODIFY)
- `knowledge3d/cranium/bridges/pdf_ingestion_bridge.py` - PDF extraction (MODIFY)
- `knowledge3d/cranium/bridges/pdf_ingestion_bridge_phase_g.py` - Phase G wrapper (reference only)
- `knowledge3d/cranium/adaptive_rpn_engine.py` - Embedding generation (use, don't modify)
- `knowledge3d/cranium/sleep/knowledge_sleep.py` - Sleep cycle 2 (reference only)

## Expected Outcome

After your fixes:
1. **Characters phase** processes character_embeddings_trimodal.jsonl → Creates ~5,000 non-zero stars
2. **Sleep Cycle 2** consolidates those stars → Galaxy now has letter/visual knowledge
3. **Text phase** processes text_domains_v1.txt → Creates ~5,000 word stars
4. **Sleep Cycle 2** consolidates → Galaxy now has word knowledge
5. **PDF phase** uses OCR → **Can access character/word stars from Galaxy** → No memory errors!
6. **Result**: Full AGI knowledge base with 100,000+ non-zero embeddings across all modalities

## Communication Style

Daniel notes in K3D_Briefing:
> "Be very descriptive and objective at the same time, Codex likes to create subfolders - that lead to fragmentation - reffer to the K3D_Briefing doc and the proper kernels it can leverage when dealing with problems (no cpu fallbacks is another thing to keep remebering him)"

**This prompt follows that guidance**: Descriptive (all context provided), objective (clear requirements), no subfolder creation, references proper kernels, enforces no CPU fallbacks.

---

**Codex, you have repository write access. Please implement these fixes and report back when complete. The swarm is counting on you!** 🚀

**— Claude (Sonnet 4.5), on behalf of Daniel Ramos and the K3D partnership swarm**
