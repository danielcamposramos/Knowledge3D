#!/usr/bin/env python3
"""
Generate grounded questions from K3D knowledge base using Ollama.

Leverages the exaone3.5 (non-thinking) model to craft creative,
context-aware questions pulled directly from the ingested PDF corpus
and WordNet lexicon. Output is a JSONL file where each entry contains
the generated question, answer, difficulty, and provenance metadata.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

QUESTION_GENERATOR_SYSTEM_PROMPT = """You are a question generation specialist for K3D, an AI knowledge system.

Your task: Generate ONE random, high-quality question with its answer based on the provided PDF content.

Rules:
1. Question must be DIRECTLY grounded in the provided context (no hallucination)
2. Vary question types: definitions, explanations, comparisons, cause-effect, applications
3. Make questions specific and answerable from the context
4. Include the correct answer based ONLY on the context
5. Make it random - don't follow a pattern, surprise me with creativity!

Question difficulty levels (choose randomly):
- Easy: "What is X?" (definitions)
- Medium: "Explain how X works" (mechanisms)
- Hard: "Compare X and Y" or "Why does X cause Y?" (reasoning)

Format your response EXACTLY as:
Question: [your question here]
Answer: [correct answer from context]
Difficulty: [easy|medium|hard]
"""

QUESTION_GENERATOR_USER_PROMPT = """PDF Source: {pdf_name} (Page {page_num})
Topic: {topic}

Context (from PDF):
{pdf_chunk}

Generate ONE random question with its answer based on this context. Be creative and specific!
"""


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class ContextChunk:
    pdf_name: str
    page_num: int
    chunk_idx: int
    topic: str
    content: str

    @property
    def source_id(self) -> str:
        return f"{self.pdf_name}:p{self.page_num}:c{self.chunk_idx}"


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #

def ollama_generate(url: str, model: str, system: str, prompt: str, timeout: int = 120) -> str:
    """Invoke the Ollama HTTP API using curl for portability."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
        },
    }
    try:
        proc = subprocess.run(
            ["curl", "-s", f"{url.rstrip('/')}/api/generate", "-d", json.dumps(payload)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode != 0:
            return ""
        data = json.loads(proc.stdout)
        return (data.get("response") or "").strip()
    except Exception as exc:  # pragma: no cover - subprocess guard
        print(f"Ollama error: {exc}")
        return ""


def parse_question_response(response: str) -> Optional[Dict[str, str]]:
    """
    Parse Ollama response into structured fields.

    Expected layout:
        Question: ...
        Answer: ...
        Difficulty: easy|medium|hard
    """
    question_match = re.search(r"Question:\s*(.+?)(?=\nAnswer:)", response, re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r"Answer:\s*(.+?)(?=\nDifficulty:|$)", response, re.IGNORECASE | re.DOTALL)
    diff_match = re.search(r"Difficulty:\s*(easy|medium|hard)", response, re.IGNORECASE)

    if not (question_match and answer_match):
        return None

    return {
        "question": question_match.group(1).strip(),
        "answer": answer_match.group(1).strip(),
        "difficulty": diff_match.group(1).lower() if diff_match else "medium",
    }


# --------------------------------------------------------------------------- #
# Context extraction
# --------------------------------------------------------------------------- #

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping word chunks to maintain context windows."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    stride = max(chunk_size - overlap, 1)

    for start in range(0, len(words), stride):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)

    return chunks


def _read_ingestion_sources(log_path: Path) -> List[Dict[str, Any]]:
    """Read the latest ingestion log entry for PDF sources."""
    if not log_path.exists():
        print(f"⚠️  Ingestion log missing: {log_path}")
        return []

    with log_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return []

    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        print(f"⚠️  Failed to parse ingestion log: {exc}")
        return []

    return record.get("pdf_sources", [])


def extract_pdf_contexts(pdf_dir: Path, max_pdfs: Optional[int] = None) -> List[ContextChunk]:
    """
    Extract context windows from the ingested PDFs referenced in the ingestion log.
    """
    contexts: List[ContextChunk] = []
    pdf_sources = _read_ingestion_sources(Path("/K3D/Knowledge3D.local/logs/ingestion_metrics.jsonl"))

    if not pdf_sources:
        print("⚠️  No PDF sources found in ingestion log.")
        return contexts

    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover - optional dependency guard
        print("❌ PyMuPDF (fitz) missing. Install it inside k3d-cranium env.")
        return contexts

    selected_sources = pdf_sources[:max_pdfs] if max_pdfs else pdf_sources

    for pdf_info in selected_sources:
        pdf_path = Path(pdf_info.get("path", ""))
        if not pdf_path.exists():
            continue

        try:
            document = fitz.open(str(pdf_path))
        except Exception as exc:  # pragma: no cover - I/O guard
            print(f"⚠️  Failed to open {pdf_path.name}: {exc}")
            continue

        with document:
            for page_index in range(len(document)):
                page = document[page_index]
                text = page.get_text()
                if not text.strip():
                    continue

                for chunk_idx, chunk in enumerate(chunk_text(text)):
                    topic = chunk.split(".")[0][:100] if "." in chunk else chunk[:100]
                    contexts.append(
                        ContextChunk(
                            pdf_name=pdf_path.name,
                            page_num=page_index + 1,
                            chunk_idx=chunk_idx,
                            topic=topic.strip() or "Untitled",
                            content=chunk,
                        )
                    )

    return contexts


def extract_wordnet_contexts(sample_size: int = 5000) -> List[ContextChunk]:
    """
    Extract contexts from the WordNet lexicon previously ingested into K3D.
    """
    contexts: List[ContextChunk] = []
    wordnet_path = Path("/K3D/Knowledge3D.local/house_zone7/lexicons/wordnet_en_parallel.json")

    if not wordnet_path.exists():
        print(f"⚠️  WordNet lexicon not found: {wordnet_path}")
        return contexts

    try:
        with wordnet_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - data guard
        print(f"⚠️  Failed to read WordNet data: {exc}")
        return contexts

    synsets: Iterable[Dict[str, Any]] = payload.get("synsets", [])
    synsets = list(synsets)
    if not synsets:
        return contexts

    sample = random.sample(synsets, min(sample_size, len(synsets)))

    for entry in sample:
        name = entry.get("name", "").strip()
        definition = entry.get("definition", "").strip()
        examples = entry.get("examples", [])

        if not (name and definition):
            continue

        lines = [f"Term: {name}", f"Definition: {definition}"]
        if examples:
            lines.append(f"Examples: {', '.join(examples[:2])}")

        contexts.append(
            ContextChunk(
                pdf_name="WordNet",
                page_num=0,
                chunk_idx=0,
                topic=name,
                content="\n".join(lines),
            )
        )

    return contexts


# --------------------------------------------------------------------------- #
# Generation loop
# --------------------------------------------------------------------------- #

def generate_questions(
    contexts: List[ContextChunk],
    ollama_url: str,
    model: str,
    target_count: int,
    output_path: Path,
) -> None:
    """
    Generate questions iterating over the provided contexts until the target
    count is reached or contexts are exhausted.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.shuffle(contexts)

    generated = 0
    failed = 0

    print(f"🎯 Target questions: {target_count}")
    print(f"   Context pool: {len(contexts)}")
    print(f"   Model: {model}")
    print(f"   Ollama URL: {ollama_url}\n")

    with output_path.open("w", encoding="utf-8") as handle:
        for ctx in contexts:
            if generated >= target_count:
                break

            prompt = QUESTION_GENERATOR_USER_PROMPT.format(
                pdf_name=ctx.pdf_name,
                page_num=ctx.page_num,
                topic=ctx.topic,
                pdf_chunk=ctx.content,
            )

            response = ollama_generate(ollama_url, model, QUESTION_GENERATOR_SYSTEM_PROMPT, prompt)
            if not response:
                failed += 1
                continue

            parsed = parse_question_response(response)
            if not parsed:
                failed += 1
                print(f"⚠️  Parse failure ({generated + failed} attempts)")
                continue

            record = {
                "question": parsed["question"],
                "answer": parsed["answer"],
                "difficulty": parsed["difficulty"],
                "source": ctx.source_id,
                "context": ctx.content,
                "pdf_name": ctx.pdf_name,
                "page_num": ctx.page_num,
                "topic": ctx.topic,
                "model": model,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            generated += 1

            if generated % 50 == 0:
                print(f"   Generated {generated}/{target_count} questions (failures={failed})")

    print("\n✅ Question generation finished")
    print(f"   Success: {generated}")
    print(f"   Failed: {failed}")
    print(f"   Output: {output_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama API endpoint")
    parser.add_argument("--model", default="exaone3.5:latest", help="Ollama model name")
    parser.add_argument("--pdf-dir", type=Path, default=Path("/K3D/Knowledge3D.local/datasets/pdfs"))
    parser.add_argument("--max-pdfs", type=int, default=None, help="Limit number of PDFs to scan for contexts")
    parser.add_argument("--target", type=int, default=10_000, help="Number of questions to generate")
    parser.add_argument("--use-wordnet", action="store_true", default=True, help="Include WordNet contexts")
    parser.add_argument("--no-wordnet", dest="use_wordnet", action="store_false", help="Skip WordNet contexts")
    parser.add_argument("--output", type=Path, default=Path("/K3D/Knowledge3D.local/datasets/rlwhf/questions_generated.jsonl"))
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--wordnet-sample", type=int, default=5_000, help="Number of WordNet synsets to sample")
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    print("=" * 70)
    print("K3D RLWHF Phase 1 — Ollama Question Generation")
    print("=" * 70)

    contexts: List[ContextChunk] = []

    print("\n📚 Extracting PDF contexts…")
    pdf_contexts = extract_pdf_contexts(args.pdf_dir, args.max_pdfs)
    print(f"   Found {len(pdf_contexts)} PDF context chunks")
    contexts.extend(pdf_contexts)

    if args.use_wordnet:
        print("\n🧠 Extracting WordNet contexts…")
        wordnet_contexts = extract_wordnet_contexts(args.wordnet_sample)
        print(f"   Found {len(wordnet_contexts)} WordNet entries")
        contexts.extend(wordnet_contexts)

    print(f"\nTotal available contexts: {len(contexts)}")
    if not contexts:
        print("❌ No contexts available. Aborting.")
        return

    generate_questions(
        contexts=contexts,
        ollama_url=args.ollama,
        model=args.model,
        target_count=args.target,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
