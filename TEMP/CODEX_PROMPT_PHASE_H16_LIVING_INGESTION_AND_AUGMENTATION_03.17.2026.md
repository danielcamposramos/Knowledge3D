# Phase H16: Living Ingestion — Benchmarks as Natural Activity + Multi-Provider Augmentation

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H15 (navigation UX) COMPLETE, existing ingestion infrastructure
**Sovereignty:** Ingestion path (flexible — Ollama, APIs allowed). Hot path remains sovereign.
**Build:** Python: `pytest`. Viewer: `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh`

---

## Paradigm Shift: Benchmarks Are Questions, Not Tasks

**Critical understanding:** K3D is an always-on system. It doesn't "run benchmarks" — it **answers questions**. ARC puzzles, math problems, GSM8K word problems, MMLU knowledge questions — these are just normal queries that humans or external systems ask.

The benchmark suite becomes a **health check**, not a mode. Like a doctor checking reflexes — the ability to answer "What is 2+3?" correctly is proof the system is healthy, not the purpose of the system.

**Sleep-time consolidation** happens AFTER answering questions:
1. System answers a batch of queries (from any source)
2. Shadow copy records what worked and what didn't
3. During idle time, sleep-time consolidation:
   - Strengthens Galaxy paths that led to correct answers
   - Weakens paths that led to incorrect answers
   - Materializes frequently-accessed patterns as House objects
   - Prunes dead-end routes

This is NOT new architecture — it's wiring the existing `SleepTimeConsolidator`, `consolidate_from_galaxy.py`, and the shadow copy system into the House-first paradigm.

---

## Deliverables

### Track A: File List Ingestion Tool (Content Scanner)
### Track B: Multi-Provider Augmentation Backend
### Track C: Ingestion → House Pipeline (Stars from Content)
### Track D: Benchmark-as-Query Integration

---

## Track A: File List Ingestion Tool (Content Scanner)

### A1. Create `knowledge3d/tools/scan_content.py`

A CLI tool that takes a file list (from `ls`, `find`, or a manifest file) and produces a content manifest ready for ingestion.

```python
"""Scan a file list and produce a content manifest for ingestion.

Usage:
  # From ls output
  ls /path/to/pdfs/*.pdf | python -m knowledge3d.tools.scan_content --output manifest.json

  # From a manifest file
  python -m knowledge3d.tools.scan_content --list files.txt --output manifest.json

  # From a directory
  python -m knowledge3d.tools.scan_content --dir /path/to/content --output manifest.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    '.pdf': 'document',
    '.txt': 'text',
    '.md': 'text',
    '.json': 'structured',
    '.jsonl': 'structured',
    '.csv': 'tabular',
    '.py': 'code',
    '.ts': 'code',
    '.js': 'code',
    '.png': 'image',
    '.jpg': 'image',
    '.jpeg': 'image',
    '.svg': 'image',
    '.mp3': 'audio',
    '.wav': 'audio',
    '.mp4': 'video',
}


def scan_file(path: Path) -> dict[str, Any] | None:
    """Produce a manifest entry for a single file."""
    if not path.exists() or not path.is_file():
        return None
    ext = path.suffix.lower()
    content_type = SUPPORTED_EXTENSIONS.get(ext)
    if not content_type:
        return None
    return {
        "path": str(path.resolve()),
        "name": path.stem,
        "extension": ext,
        "content_type": content_type,
        "size_bytes": path.stat().st_size,
        "domain_hint": guess_domain(path),
    }


def guess_domain(path: Path) -> str:
    """Heuristic domain guess from path components."""
    parts = str(path).lower()
    if 'math' in parts: return 'Mathematics'
    if 'physics' in parts: return 'Physics'
    if 'biology' in parts or 'bio' in parts: return 'Biology'
    if 'language' in parts or 'grammar' in parts: return 'Language'
    if 'tool' in parts or 'engineering' in parts: return 'Tools'
    if 'art' in parts or 'visual' in parts or 'draw' in parts: return 'Visual'
    if 'audio' in parts or 'music' in parts or 'sound' in parts: return 'Audio'
    return 'General'


def scan_content(sources: list[Path]) -> dict[str, Any]:
    """Build a complete content manifest."""
    entries = []
    for source in sources:
        entry = scan_file(source)
        if entry:
            entries.append(entry)
    return {
        "version": 1,
        "total_files": len(entries),
        "by_type": {t: sum(1 for e in entries if e["content_type"] == t) for t in set(e["content_type"] for e in entries)},
        "entries": entries,
    }
```

### A2. Create `knowledge3d/tools/ingest_from_manifest.py`

Takes a manifest from `scan_content` and runs each entry through the ingestion pipeline:

```python
"""Ingest content from a scan manifest into Galaxy entries.

Usage:
  python -m knowledge3d.tools.ingest_from_manifest \
    --manifest manifest.json \
    --provider ollama \
    --model qwen2.5:32b \
    --output /K3D/Knowledge3D.local/galaxies/ingested/
"""

import argparse
import json
from pathlib import Path

from knowledge3d.tools.augmentation_providers import create_provider
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def ingest_entry(entry: dict, provider, output_dir: Path) -> dict:
    """Ingest a single manifest entry into Galaxy-ready stars."""
    content_type = entry["content_type"]
    file_path = Path(entry["path"])

    if content_type == "document":
        return ingest_pdf(file_path, entry, provider, output_dir)
    elif content_type == "text":
        return ingest_text(file_path, entry, provider, output_dir)
    elif content_type == "structured":
        return ingest_structured(file_path, entry, provider, output_dir)
    elif content_type == "code":
        return ingest_code(file_path, entry, provider, output_dir)
    else:
        return {"status": "skipped", "reason": f"unsupported type: {content_type}"}
```

The tool reads each file, sends it to the augmentation provider for classification + enrichment, then produces `MeaningCentricStar` entries with proper `meaning_rpn`, `surface_forms`, `taxonomy_refs`, and `domain` classification.

---

## Track B: Multi-Provider Augmentation Backend

### B1. Create `knowledge3d/tools/augmentation_providers.py`

A provider abstraction that supports Ollama (local, free), Claude API (Anthropic), and GPT API (OpenAI) — all through legitimate, standard API access.

```python
"""Multi-provider augmentation backend for K3D ingestion.

Providers:
  - ollama: Local LLM via Ollama CLI (free, private, default)
  - claude: Anthropic Claude API (requires ANTHROPIC_API_KEY env var)
  - gpt: OpenAI GPT API (requires OPENAI_API_KEY env var)

All providers implement the same interface: classify + augment content
into Galaxy-ready structured data.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AugmentationResult:
    """Structured augmentation output."""
    summary: str
    entities: list[dict[str, str]]
    relationships: list[dict[str, str]]
    domain: str
    meaning_rpn_hint: str
    taxonomy_refs: list[str]
    surface_forms: dict[str, str]  # lang -> word_ref
    confidence: float
    provider: str
    raw_response: str


class AugmentationProvider(ABC):
    """Base class for augmentation providers."""

    @abstractmethod
    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        """Augment content into structured Galaxy-ready data."""
        ...

    @abstractmethod
    def classify(self, content: str) -> str:
        """Classify content into a target Galaxy domain."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""
        ...


class OllamaProvider(AugmentationProvider):
    """Local Ollama-based augmentation (default, free, private)."""

    def __init__(self, model: str = "qwen2.5:32b", timeout: float = 120.0):
        from knowledge3d.ingestion.ollama_manager import OllamaModelManager
        self.ollama = OllamaModelManager(default_timeout=timeout)
        self.model = model

    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        prompt = self._build_augmentation_prompt(content, context)
        result = self.ollama.query(self.model, prompt)
        return self._parse_result(result.output, "ollama")

    def classify(self, content: str) -> str:
        prompt = f"Classify into one domain: Mathematics, Physics, Biology, Language, Tools, Visual, Audio, General.\nContent: {content[:500]}\nDomain:"
        result = self.ollama.query(self.model, prompt)
        return result.output.strip().split('\n')[0].strip()

    def is_available(self) -> bool:
        import subprocess
        try:
            proc = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ... prompt building and parsing methods


class ClaudeProvider(AugmentationProvider):
    """Anthropic Claude API augmentation (requires ANTHROPIC_API_KEY)."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=AUGMENTATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._build_prompt(content, context)}],
        )
        return self._parse_result(message.content[0].text, "claude")

    def classify(self, content: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=64,
            messages=[{"role": "user", "content": f"Classify into one domain: Mathematics, Physics, Biology, Language, Tools, Visual, Audio, General.\nContent: {content[:500]}\nDomain:"}],
        )
        return message.content[0].text.strip().split('\n')[0].strip()

    def is_available(self) -> bool:
        return bool(self.api_key)


class GPTProvider(AugmentationProvider):
    """OpenAI GPT API augmentation (requires OPENAI_API_KEY)."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def augment(self, content: str, context: dict[str, Any]) -> AugmentationResult:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AUGMENTATION_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(content, context)},
            ],
            max_tokens=2048,
        )
        return self._parse_result(response.choices[0].message.content, "gpt")

    def classify(self, content: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"Classify: {content[:500]}\nDomain:"}],
            max_tokens=32,
        )
        return response.choices[0].message.content.strip().split('\n')[0].strip()

    def is_available(self) -> bool:
        return bool(self.api_key)


AUGMENTATION_SYSTEM_PROMPT = """You augment knowledge content for K3D Galaxy ingestion.
Return strict JSON with keys:
summary: compact summary text
entities: list of {"type","name","content"}
relationships: list of {"from","relation","to"}
domain: one of Mathematics, Physics, Biology, Language, Tools, Visual, Audio, General
meaning_rpn_hint: compact RPN-like semantic description (e.g., "KINEMATICS POSITION EULER UPDATE")
taxonomy_refs: list of related concept IDs (e.g., ["concept_mathematics", "concept_physics"])
surface_forms: {"en": "english name", "pt": "portuguese name"}
confidence: float 0-1
Avoid long narrative. Be precise and structured."""


def create_provider(name: str = "ollama", **kwargs) -> AugmentationProvider:
    """Factory for augmentation providers."""
    providers = {
        "ollama": OllamaProvider,
        "claude": ClaudeProvider,
        "gpt": GPTProvider,
    }
    cls = providers.get(name.lower())
    if not cls:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")
    return cls(**kwargs)
```

### B2. Provider priority and fallback chain

The system tries providers in order of preference:
1. **Ollama** (local, free, private) — default for all augmentation
2. **Claude API** (if `ANTHROPIC_API_KEY` set) — higher quality for complex content
3. **GPT API** (if `OPENAI_API_KEY` set) — alternative high-quality provider

The `--provider` flag selects explicitly. Without it, the system auto-detects the best available.

### B3. API access is standard and legitimate

- **Ollama**: Local CLI tool, user's own hardware. Free.
- **Claude API**: Standard Anthropic SDK (`pip install anthropic`), user's own API key from console.anthropic.com. Billed to user's account.
- **GPT API**: Standard OpenAI SDK (`pip install openai`), user's own API key from platform.openai.com. Billed to user's account.

No scraping, no unofficial access, no credential theft. Standard SDK usage with user-provided keys via environment variables.

---

## Track C: Ingestion → House Pipeline (Stars from Content)

### C1. Create `knowledge3d/tools/content_to_stars.py`

Converts augmented content into `MeaningCentricStar` entries ready for House placement:

```python
"""Convert augmented content into MeaningCentricStar entries."""

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
from knowledge3d.tools.augmentation_providers import AugmentationResult


def result_to_star(
    result: AugmentationResult,
    *,
    star_id: str,
    house_room: str | None = None,
    house_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MeaningCentricStar:
    """Convert an AugmentationResult into a MeaningCentricStar."""
    # Auto-assign room based on domain
    if house_room is None:
        house_room = DOMAIN_TO_ROOM.get(result.domain, "House/Library")

    return MeaningCentricStar(
        star_id=star_id,
        meaning_class="entry",
        meaning_rpn=result.meaning_rpn_hint or f"{result.domain.upper()} CONTENT ENTRY",
        domain=f"House/{result.domain}",
        surface_forms=_build_surface_forms(result.surface_forms),
        taxonomy_refs=result.taxonomy_refs,
        behavior_rpn="INSPECT LOAD_CONTENT",
        house_room=house_room,
        house_position=house_position,
        confidence=result.confidence,
        polarity=1,
    )


DOMAIN_TO_ROOM = {
    "Mathematics": "House/Library",
    "Physics": "House/Library",
    "Biology": "House/Garden",
    "Language": "House/Library",
    "Tools": "House/Workshop",
    "Visual": "House/Gallery",
    "Audio": "House/Gallery",
    "General": "House/Library",
}
```

### C2. Batch ingestion with House placement

The pipeline produces a `book_content_*.py`-style module for each batch:

```
Raw Files → scan_content → manifest.json
manifest.json → ingest_from_manifest → augmented entries
augmented entries → content_to_stars → MeaningCentricStar list
MeaningCentricStar list → export_house_content → house-content.json update
```

New content appears in the House automatically: as book entries (if organized as a book), as shelf items, or as Galaxy entries visible in the stellarium.

---

## Track D: Benchmark-as-Query Integration

### D1. Create `knowledge3d/tools/benchmark_health_check.py`

A lightweight script that runs benchmark questions as normal queries through the ingestion + Galaxy system, NOT as a special benchmark mode:

```python
"""Run benchmark questions as natural queries through the knowledge system.

This is NOT a benchmark runner. It's a health check that verifies
the system can answer questions naturally. Results are logged for
sleep-time consolidation.

Usage:
  python -m knowledge3d.tools.benchmark_health_check \
    --suite arc --count 10 --log health_log.jsonl
"""

import json
import time
from pathlib import Path


def run_health_check(suite: str, count: int, log_path: Path) -> dict:
    """Ask the system questions and log responses for consolidation."""
    questions = load_questions(suite, count)
    results = []
    for q in questions:
        start = time.monotonic()
        # This goes through the normal Galaxy query path,
        # NOT a special benchmark harness
        answer = query_knowledge_system(q["question"])
        elapsed = time.monotonic() - start
        correct = evaluate_answer(answer, q["expected"])
        results.append({
            "question_id": q["id"],
            "suite": suite,
            "question": q["question"],
            "answer": answer,
            "expected": q["expected"],
            "correct": correct,
            "elapsed_s": round(elapsed, 3),
            "timestamp": time.time(),
        })

    # Log for sleep-time consolidation
    with open(log_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    correct_count = sum(1 for r in results if r["correct"])
    return {
        "suite": suite,
        "total": len(results),
        "correct": correct_count,
        "score": f"{correct_count}/{len(results)}",
    }
```

### D2. Sleep-time consolidation reads health check logs

The existing `SleepTimeConsolidator` is extended to consume health check logs:

```python
# In sleep-time consolidation cycle:
# 1. Read health_log.jsonl
# 2. For each correct answer: strengthen the Galaxy path that produced it
# 3. For each incorrect answer: flag the path for review
# 4. Materialize frequently-correct patterns as House objects
# 5. Clear consumed log entries
```

This connects the circle: **Query → Answer → Log → Sleep → Consolidate → Stronger Galaxy → Better Answers**.

---

## Tips for Codex

**Tip 1 — Providers are ingestion-path only.** The Ollama/Claude/GPT providers are used ONLY during content ingestion (augmentation). They NEVER appear in the hot path (inference/reasoning). The hot path remains sovereign: PTX + Galaxy + RPN.

**Tip 2 — OllamaModelManager already exists.** Don't create a new Ollama integration. The `OllamaProvider` wraps the existing `OllamaModelManager` from `knowledge3d/ingestion/ollama_manager.py`. Same subprocess CLI pattern, same sanitization.

**Tip 3 — API SDKs are optional deps.** `anthropic` and `openai` packages are NOT required. They're imported lazily inside the provider methods. If not installed, the provider's `is_available()` returns False. Ollama is the only hard dependency.

**Tip 4 — scan_content reads from stdin or file.** Support pipe: `ls *.pdf | python -m knowledge3d.tools.scan_content`. Support file: `--list files.txt`. Support dir: `--dir /path/`. All produce the same manifest format.

**Tip 5 — AUGMENTATION_SYSTEM_PROMPT is shared.** All three providers use the same system prompt to produce the same structured JSON output. The provider abstraction means the downstream pipeline doesn't care which LLM augmented the content.

**Tip 6 — Don't touch the hot path.** This phase is entirely ingestion-path work. No changes to PTX kernels, sovereign bridges, or the composed head pipeline. Benchmarks are "health checks" that LOG results; they don't modify the inference pipeline.

**Tip 7 — Standard SDK usage.** The `anthropic` and `openai` SDKs are installed via pip. API keys are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). This is the standard, documented, legitimate way to use these APIs. No workarounds, no unofficial access.

**Tip 8 — Manifest format is simple.** The scan manifest is a JSON file with `entries[]`, each having `path`, `content_type`, `domain_hint`. It's designed to be human-readable and editable — users can manually add/remove entries before ingestion.

---

## Tests

### Python: `tests/test_scan_content.py`

```python
def test_scan_recognizes_supported_extensions(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"fake pdf")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "data.csv").write_text("a,b,c")
    manifest = scan_content([tmp_path / f for f in ["doc.pdf", "notes.txt", "data.csv"]])
    assert manifest["total_files"] == 3
    assert manifest["by_type"]["document"] == 1

def test_scan_skips_unsupported_extensions(tmp_path):
    (tmp_path / "binary.exe").write_bytes(b"\x00\x01")
    manifest = scan_content([tmp_path / "binary.exe"])
    assert manifest["total_files"] == 0

def test_domain_guess_from_path():
    assert guess_domain(Path("/books/mathematics/calculus.pdf")) == "Mathematics"
    assert guess_domain(Path("/random/stuff.txt")) == "General"
```

### Python: `tests/test_augmentation_providers.py`

```python
def test_ollama_provider_available_check():
    # Test that is_available works (may be True or False depending on env)
    provider = OllamaProvider()
    assert isinstance(provider.is_available(), bool)

def test_claude_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = ClaudeProvider()
    assert not provider.is_available()

def test_gpt_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = GPTProvider()
    assert not provider.is_available()

def test_create_provider_factory():
    provider = create_provider("ollama")
    assert isinstance(provider, OllamaProvider)
```

### Python: `tests/test_content_to_stars.py`

```python
def test_result_to_star_produces_valid_star():
    result = AugmentationResult(
        summary="Test content",
        entities=[],
        relationships=[],
        domain="Mathematics",
        meaning_rpn_hint="MATH CONTENT TEST",
        taxonomy_refs=["concept_mathematics"],
        surface_forms={"en": "Test Entry", "pt": "Entrada Teste"},
        confidence=0.9,
        provider="ollama",
        raw_response="{}",
    )
    star = result_to_star(result, star_id="test_entry_001")
    assert star.star_id == "test_entry_001"
    assert star.house_room == "House/Library"  # Math → Library
    assert "concept_mathematics" in star.taxonomy_refs
```

### Non-regression

All existing Python + viewer tests must pass. No viewer changes in this phase.

---

## Success Criteria

1. `scan_content` accepts stdin pipe, file list, or directory and produces a manifest JSON
2. `ingest_from_manifest` processes manifest entries through a provider
3. `OllamaProvider` works with existing `OllamaModelManager`
4. `ClaudeProvider` works with standard `anthropic` SDK when `ANTHROPIC_API_KEY` is set
5. `GPTProvider` works with standard `openai` SDK when `OPENAI_API_KEY` is set
6. `content_to_stars` converts augmented results to `MeaningCentricStar` entries
7. `benchmark_health_check` runs questions as normal queries and logs for consolidation
8. All existing tests pass, new tests pass
9. No hot-path changes — sovereignty preserved

---

## Files Changed/Created

| File | Action |
|------|--------|
| `knowledge3d/tools/scan_content.py` | **NEW** — File list scanner → manifest |
| `knowledge3d/tools/ingest_from_manifest.py` | **NEW** — Manifest → ingested stars |
| `knowledge3d/tools/augmentation_providers.py` | **NEW** — Multi-provider backend |
| `knowledge3d/tools/content_to_stars.py` | **NEW** — Augmented results → MeaningCentricStar |
| `knowledge3d/tools/benchmark_health_check.py` | **NEW** — Benchmark as health check |
| `tests/test_scan_content.py` | **NEW** |
| `tests/test_augmentation_providers.py` | **NEW** |
| `tests/test_content_to_stars.py` | **NEW** |

---

## Architectural Note: The Living Knowledge System

This phase embodies Daniel's vision: K3D is NOT a program you run — it's a **living system that grows**.

**The cycle:**
```
Content Sources (PDFs, text, code, media)
    ↓ scan_content
Manifest (what to ingest)
    ↓ ingest_from_manifest + provider
Augmented Entries (structured knowledge)
    ↓ content_to_stars
MeaningCentricStar entries (Galaxy-ready)
    ↓ export_house_content
House Content (visible in viewer)
    ↓ benchmark_health_check
Query Logs (what worked, what didn't)
    ↓ sleep-time consolidation
Stronger Galaxy (refined paths)
    ↓ consolidate_from_galaxy
New House Objects (materialized patterns)
    → cycle repeats
```

**Benchmarks are just questions.** The system's ability to answer them is a side effect of having good knowledge, not a goal in itself. The goal is to be a **knowledgeable entity** — and benchmarks prove it.

**Augmentation is flexible.** Ollama runs locally and privately. Claude and GPT APIs are there when higher quality is needed. The user controls which provider to use, pays their own API costs, and everything goes through standard, legitimate SDK interfaces.

**The House grows.** Every batch of ingested content becomes new stars in the Galaxy, new entries in book content, new objects materializable in the House. The viewer shows this growth in real time — more books on shelves, more stars in the stellarium, more content on the tablet.
