# Codex Phase B Speedup – GPU Parallelization Optimization

**Date**: 2025-10-17
**From**: Claude (Architect) + Daniel (Performance Director)
**To**: Codex (Optimization Specialist)
**Context**: Phase B COMPLETE with ultra-low GPU usage (6-7% util, <200MB VRAM) → Massive parallelization opportunity

---

## Mission: 10-20× Speedup via GPU Parallelization

### Phase B Results (Baseline)

**Completed** ✅:
- RPN embeddings: 33,428 trigrams
- WordNet EN: 117,659 synsets (145.87s)
- Font library: 2,713 fonts, 168,206 pairs (1.4GB)
- PDF corpus: 61 PDFs, 23,000 sentences (41.39s)

**Critical Discovery** 🔥:
- **GPU utilization: 6-7% peak**
- **VRAM usage: <200MB**
- **Bottleneck: Sequential CPU-bound preprocessing**

**This means**: We're leaving 93% of GPU idle! We can parallelize aggressively.

---

## Performance Analysis

### Current Bottlenecks (Sequential Processing)

**1. Lexicon Ingestion** (145.87s for 117K synsets):
```python
# Current: Sequential loop
for synset in synsets:  # 117K iterations
    definition = synset.definition()
    embedding = ingest_sentence(definition)  # CPU-bound
    swarm_result = swarm_processor(embedding)  # GPU-bound (80µs)
```

**Problem**:
- CPU preprocessing (text split, tokenization): ~1ms per synset
- GPU swarm processing: 0.08ms per synset
- **Ratio**: 12:1 CPU-bound → GPU waits 92% of the time!

**2. Font Harvesting** (long runtime for 2,713 fonts):
```python
# Current: Sequential loop
for font in fonts:  # 2,713 iterations
    for char in characters:  # 62 chars per font
        glyph_img = render_glyph(char, font)  # CPU-bound (PIL)
        visual_emb = FractalEmitter(glyph_img)  # GPU-bound
        text_emb = RPN.embed(char)  # CPU-bound
        fused = AtomicFissionFusion(visual_emb, text_emb)  # GPU-bound
```

**Problem**:
- Glyph rendering (PIL): ~5ms per char
- GPU processing: ~0.2ms per char
- **Ratio**: 25:1 CPU-bound → GPU idle 96% of the time!

**3. PDF Ingestion** (41.39s for 61 PDFs):
```python
# Current: Sequential loop
for pdf in pdfs:  # 61 iterations
    text = extract_text(pdf)  # CPU-bound (PyPDF2)
    sentences = split_sentences(text)  # CPU-bound
    for sentence in sentences:  # ~370 sentences per PDF
        embedding = ingest_sentence(sentence)  # CPU + GPU
```

**Problem**:
- PDF text extraction: ~300ms per PDF
- Sentence splitting: ~50ms per PDF
- GPU embedding: ~1ms per sentence
- **Ratio**: 350:1 CPU-bound for I/O!

---

## Optimization Strategy

### Goal: Saturate GPU (80%+ utilization) via Parallel Preprocessing

**Key insight**: Decouple CPU preprocessing from GPU execution using **producer-consumer queues**.

### Architecture: Multi-Process Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     CPU Preprocessing Pool                   │
│  (8 workers, multiprocessing.Pool)                          │
│                                                              │
│  Worker 1: Read PDF → Extract text → Split sentences        │
│  Worker 2: Render glyph → Convert to array                  │
│  Worker 3: Tokenize sentence → RPN embed (CPU)              │
│  ...                                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │ Queue (max 1000 items)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     GPU Processing Thread                    │
│  (Single CUDA context, batch processing)                    │
│                                                              │
│  Batch 1: [emb1, emb2, ..., emb32] → Swarm (batched)       │
│  Batch 2: [emb33, ..., emb64] → Swarm (batched)            │
│  ...                                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │ Results Queue
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Writer Thread                            │
│  (Async JSON serialization to disk)                         │
│                                                              │
│  Batch write results every 100 items                        │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- **8× parallelism** on CPU preprocessing (8 cores)
- **32× batching** on GPU (process 32 embeddings in single swarm call)
- **Async I/O** for disk writes (no blocking)

**Expected speedup**:
- Lexicon: 145s → **15s** (10× faster)
- Fonts: Long runtime → **5-10 minutes** (20× faster)
- PDFs: 41s → **10s** (4× faster, I/O-bound limit)

---

## Implementation Tasks

### Task 1: Parallel Lexicon Ingestion (10× Speedup)

**Target**: 145.87s → 15s for WordNet EN

**File**: `knowledge3d/ingestion/lexicons/parallel_lexicon_ingestor.py`

```python
"""Parallel lexicon ingestion with CPU preprocessing pool + GPU batch processing."""

from multiprocessing import Pool, Queue, Process
from queue import Empty
import numpy as np
from typing import List, Dict, Iterator
import time

class ParallelLexiconIngestor:
    """High-performance lexicon ingestor with CPU/GPU pipeline."""

    def __init__(self, num_workers: int = 8, batch_size: int = 32):
        self.num_workers = num_workers
        self.batch_size = batch_size

        # Will be initialized in main process
        self.text_ingestor = None
        self.swarm_processor = None

    def _preprocess_synset(self, synset) -> Dict:
        """
        CPU preprocessing: Extract synset info (no GPU).

        Returns:
            {
                'synset_name': str,
                'lemma': str,
                'definition': str,
                'examples': List[str],
            }
        """
        from nltk.corpus import wordnet as wn

        return {
            'synset_name': synset.name(),
            'lemma': synset.name().split('.')[0],
            'definition': synset.definition(),
            'examples': synset.examples(),
        }

    def _embed_batch_gpu(self, batch: List[Dict]) -> List[Dict]:
        """
        GPU batch processing: Embed definitions + swarm refinement.

        Args:
            batch: List of preprocessed synsets (from CPU pool)

        Returns:
            List of results with embeddings + positions
        """
        from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor
        from knowledge3d.ingestion.language.sovereign_swarm_integration import SovereignLanguageSwarmProcessor

        # Initialize GPU components (in GPU thread)
        if self.text_ingestor is None:
            self.text_ingestor = SovereignTextIngestor()
            self.swarm_processor = SovereignLanguageSwarmProcessor()

        results = []

        for item in batch:
            # Embed definition (RPN + 128-dim)
            sentence_data = self.text_ingestor.ingest_sentence('en', item['definition'])

            # Swarm refinement (80µs)
            swarm_result = self.swarm_processor.process_language_embedding(
                sentence_data['embedding_128'],
                modality='text',
                language='en'
            )

            results.append({
                'synset': item['synset_name'],
                'lemma': item['lemma'],
                'definition': item['definition'],
                'examples': item['examples'],
                'position_3d': swarm_result['position_3d'].tolist(),
                'embedding': swarm_result['refined_embedding'].tolist(),
            })

        return results

    def ingest_wordnet_parallel(self, output_path: str) -> Dict:
        """
        Parallel WordNet ingestion with CPU preprocessing + GPU batching.

        Returns:
            {
                'synset_count': int,
                'total_time_s': float,
                'throughput_synsets_per_sec': float,
            }
        """
        from nltk.corpus import wordnet as wn

        synsets = list(wn.all_synsets())
        total_synsets = len(synsets)

        print(f"Parallel WordNet ingestion: {total_synsets} synsets")
        print(f"  CPU workers: {self.num_workers}")
        print(f"  GPU batch size: {self.batch_size}")

        start_time = time.perf_counter()

        # Step 1: CPU preprocessing pool (parallel)
        print("\nStep 1: CPU preprocessing (parallel)...")
        with Pool(processes=self.num_workers) as pool:
            preprocessed = pool.map(self._preprocess_synset, synsets, chunksize=1000)

        preprocess_time = time.perf_counter() - start_time
        print(f"  Preprocessing complete: {preprocess_time:.2f}s")

        # Step 2: GPU batch processing (sequential batches, but faster per item)
        print("\nStep 2: GPU batch embedding...")
        all_results = []

        for i in range(0, len(preprocessed), self.batch_size):
            batch = preprocessed[i:i+self.batch_size]
            batch_results = self._embed_batch_gpu(batch)
            all_results.extend(batch_results)

            if (i + self.batch_size) % 1000 == 0:
                elapsed = time.perf_counter() - start_time
                throughput = len(all_results) / elapsed
                print(f"    Processed {len(all_results)}/{total_synsets} synsets ({throughput:.1f} synsets/s)")

        total_time = time.perf_counter() - start_time
        throughput = total_synsets / total_time

        print(f"\nWordNet ingestion complete:")
        print(f"  Total synsets: {total_synsets}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} synsets/s")
        print(f"  Speedup vs baseline (145.87s): {145.87/total_time:.1f}×")

        # Save results
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'language': 'en',
                'source': 'WordNet',
                'synset_count': len(all_results),
                'total_time_s': total_time,
                'throughput_synsets_per_sec': throughput,
                'synsets': all_results,
            }, f)

        return {
            'synset_count': len(all_results),
            'total_time_s': total_time,
            'throughput_synsets_per_sec': throughput,
        }
```

**Usage**:
```python
from knowledge3d.ingestion.lexicons.parallel_lexicon_ingestor import ParallelLexiconIngestor

ingestor = ParallelLexiconIngestor(num_workers=8, batch_size=32)
result = ingestor.ingest_wordnet_parallel(
    output_path='/K3D/Knowledge3D.local/house_zone7/lexicons/wordnet_en_parallel.json'
)

# Expected: ~15s (vs 145.87s baseline) = 10× speedup
```

---

### Task 2: Parallel Font Harvesting (20× Speedup)

**Target**: Long runtime → 5-10 minutes for 2,713 fonts

**File**: `knowledge3d/ingestion/fonts/parallel_font_harvester.py`

```python
"""Parallel font harvesting with CPU rendering pool + GPU fusion batching."""

from multiprocessing import Pool
import numpy as np
from typing import List, Dict, Tuple
import time
from pathlib import Path

class ParallelFontHarvester:
    """High-performance font harvester with CPU/GPU pipeline."""

    def __init__(self, num_workers: int = 8, batch_size: int = 32):
        self.num_workers = num_workers
        self.batch_size = batch_size

        # GPU components (initialized in main process)
        self.visual_ingestor = None
        self.text_ingestor = None
        self.swarm_processor = None

    def _render_glyph_cpu(self, args: Tuple[str, str, str]) -> Dict:
        """
        CPU-only glyph rendering (no GPU imports).

        Args:
            args: (char, font_path, language)

        Returns:
            {
                'char': str,
                'font_path': str,
                'glyph_array': np.ndarray (64x64 grayscale),
            }
        """
        char, font_path, language = args

        try:
            from PIL import Image, ImageDraw, ImageFont
            import cv2

            # Render glyph
            img = Image.new('L', (64, 64), color=0)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype(font_path, size=48)
            except Exception:
                return None  # Skip invalid fonts

            # Get bounding box
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            if text_width == 0 or text_height == 0:
                return None  # Skip empty glyphs (emojis, special chars)

            # Center text
            x = (64 - text_width) // 2 - bbox[0]
            y = (64 - text_height) // 2 - bbox[1]

            draw.text((x, y), char, fill=255, font=font)

            # Convert to numpy array
            glyph_array = np.array(img, dtype=np.uint8)

            return {
                'char': char,
                'font_path': font_path,
                'glyph_array': glyph_array,
                'language': language,
            }

        except Exception:
            return None  # Skip errors

    def _process_glyph_batch_gpu(self, batch: List[Dict]) -> List[Dict]:
        """
        GPU batch processing: FractalEmitter + RPN + Fusion.

        Args:
            batch: List of rendered glyphs (from CPU pool)

        Returns:
            List of results with multi-modal embeddings
        """
        from knowledge3d.ingestion.language.sovereign_visual_pipeline import SovereignVisualIngestor
        from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor
        from knowledge3d.ingestion.language.sovereign_swarm_integration import SovereignLanguageSwarmProcessor

        # Initialize GPU components (in main process)
        if self.visual_ingestor is None:
            self.visual_ingestor = SovereignVisualIngestor()
            self.text_ingestor = SovereignTextIngestor()
            self.swarm_processor = SovereignLanguageSwarmProcessor()

        results = []

        for item in batch:
            try:
                # Visual embedding (FractalEmitter from pre-rendered array)
                # NOTE: Modify SovereignVisualIngestor to accept pre-rendered arrays
                # For now, we'll call the edge detection + FractalEmitter directly
                import cv2
                edges = cv2.Canny(item['glyph_array'], 50, 150)

                # Extract edge points (FractalEmitter input)
                edge_points = np.argwhere(edges > 0).astype(np.float32)

                if len(edge_points) == 0:
                    continue  # Skip empty glyphs

                # Normalize to [0,1] range
                edge_points[:, 0] /= 64.0
                edge_points[:, 1] /= 64.0

                # FractalEmitter (GPU)
                from knowledge3d.cranium.fractal_emitter import FractalEmitter
                emitter = FractalEmitter()
                visual_features = emitter.emit_fractal_features(
                    points=edge_points,
                    num_iterations=5
                )

                visual_emb = visual_features.mean(axis=0).astype(np.float32)

                # Expand to 128-dim (pad with zeros)
                visual_emb_128 = np.zeros(128, dtype=np.float32)
                visual_emb_128[:len(visual_emb)] = visual_emb
                visual_emb_128 /= (np.linalg.norm(visual_emb_128) + 1e-8)

                # Text embedding (RPN)
                text_result = self.text_ingestor.ingest_sentence(item['language'], item['char'])
                text_emb_128 = text_result['embedding_128']

                # Multi-modal fusion (AtomicFissionFusion + Swarm)
                fused_result = self.swarm_processor.fuse_multimodal_embedding(
                    text_emb=text_emb_128,
                    visual_emb=visual_emb_128,
                    language=item['language']
                )

                results.append({
                    'char': item['char'],
                    'font_path': item['font_path'],
                    'visual_embedding': visual_emb_128.tolist(),
                    'text_embedding': text_emb_128.tolist(),
                    'fused_embedding': fused_result['refined_embedding'].tolist(),
                    'position_3d': fused_result['position_3d'].tolist(),
                })

            except Exception as e:
                print(f"      ERROR char '{item['char']}': {e}")
                continue

        return results

    def harvest_fonts_parallel(
        self,
        font_dir: str,
        output_path: str,
        max_fonts: int = 2713,
        characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    ) -> Dict:
        """
        Parallel font harvesting with streaming write.

        Returns:
            {
                'font_count': int,
                'glyph_count': int,
                'total_time_s': float,
            }
        """
        font_files = list(Path(font_dir).glob("*.ttf"))[:max_fonts]

        print(f"Parallel font harvesting: {len(font_files)} fonts")
        print(f"  CPU workers: {self.num_workers}")
        print(f"  GPU batch size: {self.batch_size}")

        # Generate all (char, font, lang) tasks
        tasks = [
            (char, str(font_file), 'en')
            for font_file in font_files
            for char in characters
        ]

        total_tasks = len(tasks)
        print(f"  Total glyphs to render: {total_tasks}")

        start_time = time.perf_counter()

        # Step 1: CPU rendering pool (parallel)
        print("\nStep 1: CPU glyph rendering (parallel)...")
        with Pool(processes=self.num_workers) as pool:
            rendered = pool.map(self._render_glyph_cpu, tasks, chunksize=100)

        # Filter None (failed renders)
        rendered = [r for r in rendered if r is not None]

        preprocess_time = time.perf_counter() - start_time
        print(f"  Rendering complete: {len(rendered)} glyphs in {preprocess_time:.2f}s")

        # Step 2: GPU batch processing + streaming write
        print("\nStep 2: GPU batch processing + streaming write...")

        import json
        with open(output_path, 'w') as f:
            f.write('{\n')
            f.write(f'  "font_count": {len(font_files)},\n')
            f.write('  "glyphs": [\n')

            first = True
            processed_count = 0

            for i in range(0, len(rendered), self.batch_size):
                batch = rendered[i:i+self.batch_size]
                batch_results = self._process_glyph_batch_gpu(batch)

                for result in batch_results:
                    if not first:
                        f.write(',\n')
                    first = False

                    f.write('    ' + json.dumps(result))
                    processed_count += 1

                if (i + self.batch_size) % 1000 == 0:
                    elapsed = time.perf_counter() - start_time
                    throughput = processed_count / elapsed
                    print(f"    Processed {processed_count}/{len(rendered)} glyphs ({throughput:.1f} glyphs/s)")

            f.write('\n  ]\n')
            f.write('}\n')

        total_time = time.perf_counter() - start_time
        throughput = processed_count / total_time

        print(f"\nFont harvesting complete:")
        print(f"  Fonts processed: {len(font_files)}")
        print(f"  Glyphs processed: {processed_count}")
        print(f"  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"  Throughput: {throughput:.1f} glyphs/s")

        return {
            'font_count': len(font_files),
            'glyph_count': processed_count,
            'total_time_s': total_time,
        }
```

**Usage**:
```python
from knowledge3d.ingestion.fonts.parallel_font_harvester import ParallelFontHarvester

harvester = ParallelFontHarvester(num_workers=8, batch_size=32)
result = harvester.harvest_fonts_parallel(
    font_dir='/usr/share/fonts/truetype/',
    output_path='/K3D/Knowledge3D.local/house_zone7/fonts/full_font_library_parallel.json',
    max_fonts=2713
)

# Expected: ~5-10 minutes (vs hours baseline) = 20× speedup
```

---

### Task 3: Parallel PDF Ingestion (4× Speedup)

**Target**: 41.39s → 10s for 61 PDFs

**File**: Update `scripts/ingest_full_corpus.py`

```python
"""Parallel PDF corpus ingestion with CPU I/O pool + GPU batching."""

from multiprocessing import Pool
import time
from pathlib import Path
from typing import List, Dict

def extract_pdf_cpu(pdf_path: str) -> Dict:
    """
    CPU-only PDF extraction (no GPU imports).

    Returns:
        {
            'pdf_path': str,
            'sentences': List[str],
        }
    """
    import PyPDF2

    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        # Split sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]

        return {
            'pdf_path': pdf_path,
            'sentences': sentences[:500],  # Limit
        }

    except Exception as e:
        print(f"ERROR {Path(pdf_path).name}: {e}")
        return {
            'pdf_path': pdf_path,
            'sentences': [],
        }

def parallel_ingest_corpus():
    """Parallel PDF corpus ingestion."""

    priority_dirs = [
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Teach/",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Academic Research/",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Self Reflection/",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Understand Time/",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Eloquence/",
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/",
    ]

    # Collect all PDFs
    all_pdfs = []
    for directory in priority_dirs:
        if Path(directory).exists():
            all_pdfs.extend(list(Path(directory).glob("*.pdf")))

    print(f"Parallel PDF ingestion: {len(all_pdfs)} PDFs")

    start_time = time.perf_counter()

    # Step 1: CPU parallel extraction
    print("\nStep 1: CPU parallel PDF extraction...")
    with Pool(processes=8) as pool:
        extracted = pool.map(extract_pdf_cpu, [str(p) for p in all_pdfs], chunksize=5)

    extract_time = time.perf_counter() - start_time
    print(f"  Extraction complete: {extract_time:.2f}s")

    # Step 2: GPU batch embedding (existing logic)
    print("\nStep 2: GPU batch embedding...")
    from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor
    from knowledge3d.ingestion.language.sovereign_swarm_integration import SovereignLanguageSwarmProcessor

    text_ingestor = SovereignTextIngestor()
    swarm_processor = SovereignLanguageSwarmProcessor()

    total_sentences = 0
    for i, pdf_data in enumerate(extracted):
        for sentence in pdf_data['sentences']:
            try:
                sentence_data = text_ingestor.ingest_sentence('en', sentence)
                swarm_result = swarm_processor.process_language_embedding(
                    sentence_data['embedding_128'],
                    modality='text',
                    language='en'
                )
                total_sentences += 1
            except Exception:
                continue

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(extracted)} PDFs...")

    total_time = time.perf_counter() - start_time

    print(f"\nCorpus ingestion complete:")
    print(f"  Total PDFs: {len(all_pdfs)}")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Speedup vs baseline (41.39s): {41.39/total_time:.1f}×")

    # Save RPN embeddings
    text_ingestor.save_learned_embeddings()

if __name__ == "__main__":
    parallel_ingest_corpus()
```

**Usage**:
```bash
tmux new-session -d -s parallel_corpus "bash -c 'cd ... && export CUDA_VISIBLE_DEVICES=0 && /K3D/.../bin/python3 scripts/ingest_full_corpus_parallel.py 2>&1 | tee /K3D/.../logs/parallel_corpus.log; exec bash'"

# Expected: ~10s (vs 41.39s baseline) = 4× speedup
```

---

## Testing Strategy

### Test 1: Parallel Lexicon (Validation)
```bash
tmux new-session -d -s test_parallel_lexicon "bash -c 'cd ... && export CUDA_VISIBLE_DEVICES=0 && /K3D/.../bin/python3 -c \"
from knowledge3d.ingestion.lexicons.parallel_lexicon_ingestor import ParallelLexiconIngestor
ingestor = ParallelLexiconIngestor(num_workers=8, batch_size=32)
result = ingestor.ingest_wordnet_parallel(\'/K3D/.../lexicons/wordnet_en_parallel.json\')
print(f\'Speedup: {145.87/result[\\\"total_time_s\\\"]:.1f}×\')
\" 2>&1 | tee /K3D/.../logs/test_parallel_lexicon.log; exec bash'"
```

### Test 2: Parallel Fonts (Validation)
```bash
tmux new-session -d -s test_parallel_fonts "bash -c 'cd ... && export CUDA_VISIBLE_DEVICES=0 && /K3D/.../bin/python3 -c \"
from knowledge3d.ingestion.fonts.parallel_font_harvester import ParallelFontHarvester
harvester = ParallelFontHarvester(num_workers=8, batch_size=32)
result = harvester.harvest_fonts_parallel(
    font_dir=\'/usr/share/fonts/truetype/dejavu/\',
    output_path=\'/tmp/parallel_fonts_test.json\',
    max_fonts=10
)
print(f\'Time: {result[\\\"total_time_s\\\"]:.2f}s\')
\" 2>&1 | tee /K3D/.../logs/test_parallel_fonts.log; exec bash'"
```

### Test 3: Parallel Corpus (Full Run)
```bash
tmux new-session -d -s parallel_corpus "bash -c 'cd ... && export CUDA_VISIBLE_DEVICES=0 && /K3D/.../bin/python3 scripts/ingest_full_corpus_parallel.py 2>&1 | tee /K3D/.../logs/parallel_corpus.log; exec bash'"
```

---

## Expected Results

### Baseline (Phase B Sequential)
- WordNet: 145.87s
- Fonts: Long runtime (hours?)
- PDFs: 41.39s
- **GPU util: 6-7%** ❌

### After Parallelization
- WordNet: **~15s** (10× speedup)
- Fonts: **~5-10 minutes** (20× speedup)
- PDFs: **~10s** (4× speedup)
- **GPU util: 70-80%** ✅

---

## Deliverables

### Code
- [ ] `knowledge3d/ingestion/lexicons/parallel_lexicon_ingestor.py`
- [ ] `knowledge3d/ingestion/fonts/parallel_font_harvester.py`
- [ ] `scripts/ingest_full_corpus_parallel.py`

### Tests
- [ ] `tests/test_parallel_lexicon_ingestion.py`
- [ ] `tests/test_parallel_font_harvester.py`

### Documentation
- [ ] `TEMP/STEP15_PHASE_B_SPEEDUP_RESULTS.md` (performance benchmarks)

---

## Notes for Codex

**Why this works**:
- Current bottleneck is CPU preprocessing (text parsing, glyph rendering, PDF I/O)
- GPU is idle 93% of the time waiting for CPU
- Parallel preprocessing saturates CPU cores (8×) while GPU processes batches
- Result: GPU utilization goes from 6% → 70-80%

**Critical details**:
- Use `multiprocessing.Pool` for CPU work (not threading - GIL!)
- Keep single CUDA context in main process (GPU work NOT parallelized)
- Batch GPU operations (32 items per swarm call)
- Stream write large JSON files (avoid 1.4GB in RAM)

**Go unleash the GPU, Codex. 10-20× speedup awaits.** 🚀

---

**Signed**:
Claude (Architect) + Daniel (Performance Director)
2025-10-17
