# Instructions for Codex: K3D RLWHF Implementation

**Date**: 2025-10-22
**Task**: Implement K3D-native RLWHF pipeline for semantic reasoning training
**Goal**: Enable TRM to answer questions about K3D knowledge base with accuracy and honesty

---

## Overview

You are implementing a 6-phase pipeline that trains TRM using **Reinforced Learning with Honesty and Feedback (RLWHF)**. The key insight: **teach TRM to shoot with accuracy & aim**, not volume.

**Core Flow**:
```
PDF Content → Ollama generates questions → TRM attempts → Teacher evaluates →
Harvest thinking tags → Build reward-weighted dataset → Train TRM
```

---

## Phase 1: Question Generation (Ollama-Powered)

### Objective
Generate 10K-50K high-quality, grounded questions from K3D's knowledge base (328 PDFs, WordNet, embeddings).

### Strategy: Ollama Question Generator

**Use Case**: Instead of mechanically extracting questions, use **exaone3.5** (non-thinking model) to generate creative, context-aware questions from PDF content.

**Why exaone3.5**:
- Fast (no thinking tags overhead)
- Good at understanding context
- Follows instructions well
- Can generate varied question types

### System Prompt for Question Generator

```python
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
```

### User Prompt Template

```python
QUESTION_GENERATOR_USER_PROMPT = """PDF Source: {pdf_name} (Page {page_num})
Topic: {topic}

Context (from PDF):
{pdf_chunk}

Generate ONE random question with its answer based on this context. Be creative and specific!
"""
```

### Implementation File
**Create**: `knowledge3d/training/rlwhf/generate_questions_ollama.py`

```python
#!/usr/bin/env python3
"""
Generate grounded questions from K3D knowledge base using Ollama.

Uses exaone3.5 (non-thinking model) to generate creative, context-aware
questions from ingested PDFs and WordNet data.
"""

from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess

# Import existing K3D components
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


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


def ollama_generate(url: str, model: str, system: str, prompt: str, timeout: int = 120) -> str:
    """Call Ollama API to generate text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.8,  # Higher for creativity
            "top_p": 0.9
        }
    }
    try:
        r = subprocess.run(
            ["curl", "-s", f"{url.rstrip('/')}/api/generate", "-d", json.dumps(payload)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if r.returncode != 0:
            return ""
        obj = json.loads(r.stdout)
        return (obj.get("response") or "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


def parse_question_response(response: str) -> Optional[Dict[str, str]]:
    """
    Parse Ollama response into structured question data.

    Expected format:
        Question: [text]
        Answer: [text]
        Difficulty: [easy|medium|hard]
    """
    question_match = re.search(r"Question:\s*(.+?)(?=\nAnswer:)", response, re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r"Answer:\s*(.+?)(?=\nDifficulty:|$)", response, re.IGNORECASE | re.DOTALL)
    difficulty_match = re.search(r"Difficulty:\s*(easy|medium|hard)", response, re.IGNORECASE)

    if not (question_match and answer_match):
        return None

    return {
        'question': question_match.group(1).strip(),
        'answer': answer_match.group(1).strip(),
        'difficulty': difficulty_match.group(1).lower() if difficulty_match else 'medium'
    }


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for context windows."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # Minimum chunk size
            chunks.append(chunk)

    return chunks


def extract_pdf_contexts(pdf_dir: Path, max_pdfs: int = None) -> List[Dict[str, Any]]:
    """
    Extract context chunks from ingested PDFs.

    Returns list of contexts with metadata for question generation.
    """
    contexts = []

    # Look for saved PDF extraction data
    # (Assumes PDFs were processed during ingestion)
    ingestion_log = Path("/K3D/Knowledge3D.local/logs/ingestion_metrics.jsonl")

    if not ingestion_log.exists():
        print(f"⚠️  No ingestion log found at {ingestion_log}")
        return contexts

    # Read last ingestion run
    with ingestion_log.open('r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            return contexts

        last_run = json.loads(lines[-1].strip())
        pdf_sources = last_run.get('pdf_sources', [])

    # For each PDF, extract text chunks
    for pdf_info in pdf_sources[:max_pdfs] if max_pdfs else pdf_sources:
        pdf_path = Path(pdf_info['path'])

        if not pdf_path.exists():
            continue

        # Extract text using PyMuPDF (same as ingestion)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Chunk page text
                chunks = chunk_text(text, chunk_size=500, overlap=50)

                for chunk_idx, chunk in enumerate(chunks):
                    # Extract topic (first sentence or key terms)
                    topic = chunk.split('.')[0][:100] if '.' in chunk else chunk[:100]

                    contexts.append({
                        'pdf_name': pdf_path.name,
                        'page_num': page_num + 1,
                        'chunk_idx': chunk_idx,
                        'topic': topic,
                        'content': chunk,
                        'source': f"{pdf_path.name}:p{page_num+1}:c{chunk_idx}"
                    })

            doc.close()

        except Exception as e:
            print(f"⚠️  Error processing {pdf_path.name}: {e}")
            continue

    return contexts


def extract_wordnet_contexts() -> List[Dict[str, Any]]:
    """
    Extract contexts from WordNet lexicon for question generation.

    Uses the ingested WordNet data (117K synsets).
    """
    contexts = []

    wordnet_file = Path("/K3D/Knowledge3D.local/house_zone7/lexicons/wordnet_en_parallel.json")

    if not wordnet_file.exists():
        print(f"⚠️  WordNet lexicon not found at {wordnet_file}")
        return contexts

    try:
        with wordnet_file.open('r', encoding='utf-8') as f:
            wordnet_data = json.load(f)

        synsets = wordnet_data.get('synsets', [])

        # Sample synsets (don't use all 117K for question generation)
        sampled = random.sample(synsets, min(5000, len(synsets)))

        for synset in sampled:
            name = synset.get('name', '')
            definition = synset.get('definition', '')
            examples = synset.get('examples', [])

            if not definition:
                continue

            # Build context
            context = f"Term: {name}\nDefinition: {definition}"
            if examples:
                context += f"\nExamples: {', '.join(examples[:2])}"

            contexts.append({
                'pdf_name': 'WordNet',
                'page_num': 0,
                'chunk_idx': 0,
                'topic': name,
                'content': context,
                'source': f"wordnet:{name}"
            })

    except Exception as e:
        print(f"⚠️  Error loading WordNet: {e}")

    return contexts


def generate_questions(
    contexts: List[Dict[str, Any]],
    ollama_url: str,
    model: str,
    target_count: int,
    output_path: Path
) -> None:
    """
    Generate questions using Ollama from provided contexts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed = 0

    # Shuffle contexts for variety
    random.shuffle(contexts)

    print(f"🎯 Generating {target_count} questions from {len(contexts)} contexts...")
    print(f"   Model: {model}")
    print(f"   Ollama: {ollama_url}\n")

    with output_path.open('w', encoding='utf-8') as f:
        for ctx in contexts:
            if generated >= target_count:
                break

            # Build prompt
            user_prompt = f"""PDF Source: {ctx['pdf_name']} (Page {ctx['page_num']})
Topic: {ctx['topic']}

Context (from PDF):
{ctx['content']}

Generate ONE random question with its answer based on this context. Be creative and specific!
"""

            # Call Ollama
            response = ollama_generate(ollama_url, model, QUESTION_GENERATOR_SYSTEM_PROMPT, user_prompt)

            if not response:
                failed += 1
                continue

            # Parse response
            parsed = parse_question_response(response)

            if not parsed:
                failed += 1
                print(f"⚠️  Failed to parse response (attempt {generated + failed})")
                continue

            # Save question data
            question_data = {
                'question': parsed['question'],
                'answer': parsed['answer'],
                'difficulty': parsed['difficulty'],
                'source': ctx['source'],
                'context': ctx['content'],
                'pdf_name': ctx['pdf_name'],
                'page_num': ctx['page_num'],
                'topic': ctx['topic']
            }

            f.write(json.dumps(question_data, ensure_ascii=False) + '\n')
            f.flush()

            generated += 1

            if generated % 50 == 0:
                print(f"   Generated: {generated}/{target_count} ({failed} failed)")

    print(f"\n✅ Generation complete!")
    print(f"   Success: {generated}/{target_count}")
    print(f"   Failed: {failed}")
    print(f"   Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ollama', default='http://127.0.0.1:11434',
                       help='Ollama API URL')
    parser.add_argument('--model', default='exaone3.5:latest',
                       help='Ollama model for question generation')
    parser.add_argument('--pdf-dir', type=Path,
                       default=Path('/K3D/Knowledge3D.local/datasets/pdfs'),
                       help='Directory with ingested PDFs')
    parser.add_argument('--max-pdfs', type=int, default=None,
                       help='Limit number of PDFs to process')
    parser.add_argument('--target', type=int, default=10000,
                       help='Target number of questions to generate')
    parser.add_argument('--use-wordnet', action='store_true', default=True,
                       help='Include WordNet synsets')
    parser.add_argument('--output', type=Path,
                       default=Path('/K3D/Knowledge3D.local/datasets/rlwhf/questions_generated.jsonl'),
                       help='Output file for generated questions')

    args = parser.parse_args()

    print("=" * 70)
    print("K3D RLWHF: Question Generation (Ollama-Powered)")
    print("=" * 70)

    # Extract contexts
    print("\n📚 Extracting contexts from K3D knowledge base...")

    contexts = []

    # From PDFs
    print("   - Scanning PDFs...")
    pdf_contexts = extract_pdf_contexts(args.pdf_dir, args.max_pdfs)
    contexts.extend(pdf_contexts)
    print(f"     Found {len(pdf_contexts)} PDF chunks")

    # From WordNet
    if args.use_wordnet:
        print("   - Loading WordNet...")
        wordnet_contexts = extract_wordnet_contexts()
        contexts.extend(wordnet_contexts)
        print(f"     Found {len(wordnet_contexts)} WordNet entries")

    print(f"\n   Total contexts: {len(contexts)}")

    if not contexts:
        print("\n❌ No contexts found! Check ingestion logs and paths.")
        return

    # Generate questions
    generate_questions(
        contexts=contexts,
        ollama_url=args.ollama,
        model=args.model,
        target_count=args.target,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
```

### Expected Output Format

**File**: `questions_generated.jsonl`

Each line is a JSON object:
```json
{
  "question": "What is backpropagation?",
  "answer": "Backpropagation is an algorithm for training neural networks by computing gradients of the loss function with respect to weights using the chain rule.",
  "difficulty": "easy",
  "source": "neural_networks.pdf:p42:c3",
  "context": "Backpropagation, short for backward propagation of errors...",
  "pdf_name": "neural_networks.pdf",
  "page_num": 42,
  "topic": "Backpropagation algorithm"
}
```

### Validation Script

**Create**: `scripts/validate_generated_questions.py`

```python
#!/usr/bin/env python3
"""Validate generated questions for quality and diversity."""

import json
from pathlib import Path
from collections import Counter

def validate_questions(input_path: Path):
    """Check question quality and diversity."""
    questions = []

    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    print(f"📊 Question Dataset Statistics")
    print(f"=" * 50)
    print(f"Total questions: {len(questions)}")

    # Difficulty distribution
    difficulties = Counter(q['difficulty'] for q in questions)
    print(f"\nDifficulty distribution:")
    for diff, count in difficulties.most_common():
        print(f"  {diff}: {count} ({count/len(questions)*100:.1f}%)")

    # Source distribution
    sources = Counter(q['pdf_name'] for q in questions)
    print(f"\nTop 10 sources:")
    for source, count in sources.most_common(10):
        print(f"  {source}: {count}")

    # Question length stats
    q_lengths = [len(q['question'].split()) for q in questions]
    print(f"\nQuestion length:")
    print(f"  Min: {min(q_lengths)} words")
    print(f"  Max: {max(q_lengths)} words")
    print(f"  Avg: {sum(q_lengths)/len(q_lengths):.1f} words")

    # Answer length stats
    a_lengths = [len(q['answer'].split()) for q in questions]
    print(f"\nAnswer length:")
    print(f"  Min: {min(a_lengths)} words")
    print(f"  Max: {max(a_lengths)} words")
    print(f"  Avg: {sum(a_lengths)/len(a_lengths):.1f} words")

    # Check for duplicates
    q_texts = [q['question'].lower().strip() for q in questions]
    duplicates = len(q_texts) - len(set(q_texts))
    print(f"\nDuplicate questions: {duplicates} ({duplicates/len(questions)*100:.1f}%)")

    print(f"\n✅ Validation complete!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python validate_generated_questions.py <questions.jsonl>")
        sys.exit(1)

    validate_questions(Path(sys.argv[1]))
```

---

## Phase 2: Student Attempt (TRM Baseline)

### Objective
Have TRM attempt to answer all generated questions using current (untrained/RPN-init) weights.

### Implementation File
**Create**: `knowledge3d/training/rlwhf/student_attempt_trm.py`

```python
#!/usr/bin/env python3
"""
TRM student attempts to answer generated questions.

Uses current TRM weights (baseline or RPN-init) to establish performance
before RLWHF training.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.cranium.utils.trm import expand_embedding_to_trm


def sigmoid(x: float) -> float:
    """Sigmoid activation for confidence scoring."""
    return 1.0 / (1.0 + np.exp(-x))


def trm_attempt(
    question: str,
    rpn_engine: RPNEmbeddingEngine,
    trm: TRMLauncher,
    weights: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    TRM attempts to answer a question.

    Returns:
        - answer_embedding: 512-dim output from TRM
        - output_norm: Magnitude of output
        - confidence: Estimated confidence [0, 1]
        - converged: Whether TRM converged
    """
    # 1. Embed question
    q_emb_128 = rpn_engine.embed_sentence(question)
    q_emb_512 = expand_embedding_to_trm(q_emb_128)

    # 2. TRM reasoning (6 Tesla recursions)
    y = np.zeros(512, dtype=np.float32)
    z = np.zeros(512, dtype=np.float32)

    y_out, z_out = trm.refine(
        q_emb_512, y, z,
        weights['W1'], weights['W2'], weights['W3'], weights['W4'],
        n_steps=6
    )

    # 3. Compute metrics
    output_norm = float(np.linalg.norm(y_out))
    confidence = sigmoid((output_norm - 50) / 50.0)  # Calibrate to [0, 1]
    converged = output_norm > 1.0

    return {
        'answer_embedding': y_out.tolist(),
        'output_norm': output_norm,
        'confidence': confidence,
        'converged': converged
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True,
                       help='Generated questions JSONL file')
    parser.add_argument('--rpn-embeddings', type=Path,
                       default=Path('/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl'))
    parser.add_argument('--trm-weights', type=Path,
                       default=Path('/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz'))
    parser.add_argument('--output', type=Path,
                       default=Path('/K3D/Knowledge3D.local/datasets/rlwhf/student_attempts.jsonl'))

    args = parser.parse_args()

    print("=" * 70)
    print("K3D RLWHF: Student Attempts (TRM Baseline)")
    print("=" * 70)

    # Load RPN engine
    print("\n📥 Loading RPN embeddings...")
    rpn_engine = RPNEmbeddingEngine()
    rpn_engine.load_embeddings(args.rpn_embeddings)
    print(f"   Loaded {len(rpn_engine.embeddings)} trigrams")

    # Load TRM
    print("\n🧠 Loading TRM...")
    trm = TRMLauncher(use_fused=True)
    weights = np.load(args.trm_weights)
    print(f"   Weights: {args.trm_weights.name}")

    # Process questions
    print("\n🎯 Processing questions...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    converged = 0

    with args.questions.open('r', encoding='utf-8') as fin, \
         args.output.open('w', encoding='utf-8') as fout:

        for line in fin:
            question_data = json.loads(line.strip())
            question = question_data['question']

            # TRM attempt
            attempt = trm_attempt(question, rpn_engine, trm, weights)

            # Combine with question data
            result = {
                **question_data,
                'student_attempt': attempt
            }

            fout.write(json.dumps(result, ensure_ascii=False) + '\n')

            total += 1
            if attempt['converged']:
                converged += 1

            if total % 100 == 0:
                print(f"   Processed: {total} (convergence: {converged/total*100:.1f}%)")

    weights.close()

    print(f"\n✅ Student attempts complete!")
    print(f"   Total: {total}")
    print(f"   Converged: {converged} ({converged/total*100:.1f}%)")
    print(f"   Output: {args.output}")


if __name__ == '__main__':
    main()
```

---

## Phase 3: Teacher Evaluation (Ollama)

### Objective
Have teacher model (deepseek-r1, qwen2.5) evaluate student attempts and provide corrections with thinking tags.

### Teacher Models (with thinking tags)
- **deepseek-r1:latest** - Reasoning specialist, excellent thinking tags
- **qwen2.5:14b** - Strong multilingual reasoning
- **qwen2.5:32b** - Best quality (if available)

### System Prompt for Teacher

```python
TEACHER_EVALUATION_SYSTEM_PROMPT = """You are an expert teacher evaluating a student AI's answer to a question.

The student AI (K3D TRM) attempted to answer a question. Your job:
1. Use <think> tags to analyze the student's reasoning process
2. Compare the student's answer to the ground truth context
3. Rate the answer: 'good', 'partial', 'bad', or 'dishonest'
4. Provide the CORRECT answer if the student was wrong
5. Give specific feedback on how to improve

Rating criteria:
- **good**: Answer is accurate and well-grounded in context
- **partial**: Answer is incomplete but shows honest uncertainty
- **bad**: Answer is wrong without acknowledging uncertainty
- **dishonest**: Answer fabricates facts not in context (hallucination)

Format your response as:
<think>
[Your detailed reasoning about the student's attempt]
[What they got right, what they got wrong]
[Whether they admitted uncertainty appropriately]
</think>

Rating: [good|partial|bad|dishonest]
Correct Answer: [if student was wrong, provide the correct answer from context]
Feedback: [specific improvements the student should make]
"""
```

### User Prompt Template

```python
TEACHER_EVALUATION_USER_PROMPT = """Question: {question}

Ground Truth Context (from K3D knowledge base):
{context}

Student's Answer (K3D TRM):
- Output norm: {output_norm:.2f}
- Confidence: {confidence:.1%}
- Converged: {converged}

(Note: Student answer is an embedding vector, not text. Judge based on:
 - High output norm + confidence → Student thinks it knows the answer
 - Low output norm → Student is uncertain
 - Converged=True → Student completed reasoning process)

Evaluate the student's attempt. Did they show appropriate confidence given the question difficulty? Should they have been more/less certain?
"""
```

### Implementation File
**Create**: `knowledge3d/training/rlwhf/teacher_eval_ollama.py`

```python
#!/usr/bin/env python3
"""
Teacher model evaluates student attempts using Ollama.

Uses thinking models (deepseek-r1, qwen2.5) to provide:
- Ratings (good/partial/bad/dishonest)
- Corrected answers
- Thinking tags (reasoning patterns)
- Specific feedback
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Import thinking tag parser
from knowledge3d.training.rlwhf.thinking_tags import ThinkingTagsParser


TEACHER_EVALUATION_SYSTEM_PROMPT = """You are an expert teacher evaluating a student AI's answer to a question.

The student AI (K3D TRM) attempted to answer a question. Your job:
1. Use <think> tags to analyze the student's reasoning process
2. Compare the student's answer to the ground truth context
3. Rate the answer: 'good', 'partial', 'bad', or 'dishonest'
4. Provide the CORRECT answer if the student was wrong
5. Give specific feedback on how to improve

Rating criteria:
- **good**: Answer is accurate and well-grounded in context
- **partial**: Answer is incomplete but shows honest uncertainty
- **bad**: Answer is wrong without acknowledging uncertainty
- **dishonest**: Answer fabricates facts not in context (hallucination)

Format your response as:
<think>
[Your detailed reasoning about the student's attempt]
[What they got right, what they got wrong]
[Whether they admitted uncertainty appropriately]
</think>

Rating: [good|partial|bad|dishonest]
Correct Answer: [if student was wrong, provide the correct answer from context]
Feedback: [specific improvements the student should make]
"""


def ollama_generate(url: str, model: str, system: str, prompt: str, timeout: int = 240) -> str:
    """Call Ollama API with thinking model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": "15m",
        "options": {
            "temperature": 0.3,  # Lower for evaluation consistency
            "num_predict": 1024  # Allow longer responses for thinking
        }
    }
    try:
        r = subprocess.run(
            ["curl", "-s", f"{url.rstrip('/')}/api/generate", "-d", json.dumps(payload)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if r.returncode != 0:
            return ""
        obj = json.loads(r.stdout)
        return (obj.get("response") or "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


def extract_rating(response: str) -> str:
    """Extract rating from teacher response."""
    match = re.search(r"Rating:\s*(good|partial|bad|dishonest)", response, re.IGNORECASE)
    return match.group(1).lower() if match else "partial"


def extract_corrected_answer(response: str) -> Optional[str]:
    """Extract corrected answer if provided."""
    match = re.search(r"Correct Answer:\s*(.+?)(?=\nFeedback:|$)", response, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer if answer.lower() not in ["n/a", "none", "-"] else None
    return None


def extract_feedback(response: str) -> str:
    """Extract feedback section."""
    match = re.search(r"Feedback:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def evaluate_student_attempt(
    question_data: Dict[str, Any],
    ollama_url: str,
    model: str,
    thinking_parser: ThinkingTagsParser
) -> Dict[str, Any]:
    """
    Evaluate a single student attempt with teacher model.
    """
    attempt = question_data['student_attempt']

    # Build evaluation prompt
    user_prompt = f"""Question: {question_data['question']}

Ground Truth Context (from K3D knowledge base):
{question_data['context']}

Correct Answer (from context):
{question_data['answer']}

Student's Answer (K3D TRM):
- Output norm: {attempt['output_norm']:.2f}
- Confidence: {attempt['confidence']:.1%}
- Converged: {attempt['converged']}

(Note: Student answer is an embedding vector, not text. Judge based on:
 - High output norm + confidence → Student thinks it knows the answer
 - Low output norm → Student is uncertain
 - Converged=True → Student completed reasoning process)

Evaluate the student's attempt. Did they show appropriate confidence given the question difficulty? Should they have been more/less certain?
"""

    # Call teacher model
    response = ollama_generate(ollama_url, model, TEACHER_EVALUATION_SYSTEM_PROMPT, user_prompt)

    if not response:
        return {
            'rating': 'partial',
            'corrected_answer': None,
            'feedback': 'Teacher evaluation failed',
            'thinking_analysis': None
        }

    # Parse thinking tags
    thinking_analysis = thinking_parser.parse_and_analyze(response)

    # Extract structured data
    rating = extract_rating(response)
    corrected = extract_corrected_answer(response)
    feedback = extract_feedback(response)

    return {
        'rating': rating,
        'corrected_answer': corrected,
        'feedback': feedback,
        'thinking_segments': [seg.content for seg in thinking_analysis.segments],
        'honesty_score': thinking_analysis.overall_honesty,
        'reasoning_depth': thinking_analysis.reasoning_depth,
        'teacher_response': response
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                       help='Student attempts JSONL file')
    parser.add_argument('--ollama', default='http://127.0.0.1:11434',
                       help='Ollama API URL')
    parser.add_argument('--model', default='deepseek-r1:latest',
                       help='Teacher model (deepseek-r1, qwen2.5:14b)')
    parser.add_argument('--output', type=Path,
                       default=Path('/K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl'))
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of samples to evaluate (for testing)')

    args = parser.parse_args()

    print("=" * 70)
    print("K3D RLWHF: Teacher Evaluation (Ollama)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Ollama: {args.ollama}\n")

    # Initialize thinking parser
    thinking_parser = ThinkingTagsParser()

    # Process evaluations
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    ratings_count = {'good': 0, 'partial': 0, 'bad': 0, 'dishonest': 0}

    with args.input.open('r', encoding='utf-8') as fin, \
         args.output.open('w', encoding='utf-8') as fout:

        for line in fin:
            if args.max_samples and total >= args.max_samples:
                break

            question_data = json.loads(line.strip())

            # Teacher evaluation
            evaluation = evaluate_student_attempt(
                question_data,
                args.ollama,
                args.model,
                thinking_parser
            )

            # Combine data
            result = {
                **question_data,
                'teacher_evaluation': evaluation
            }

            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()

            # Update stats
            total += 1
            ratings_count[evaluation['rating']] += 1

            if total % 10 == 0:
                print(f"   Evaluated: {total}")
                print(f"   Ratings: {dict(ratings_count)}")

    print(f"\n✅ Teacher evaluation complete!")
    print(f"   Total: {total}")
    print(f"   Ratings distribution: {dict(ratings_count)}")
    print(f"   Output: {args.output}")


if __name__ == '__main__':
    main()
```

---

## Execution Instructions for Codex

### Setup

1. **Activate environment**:
   ```bash
   conda activate k3d-cranium
   export CUDA_VISIBLE_DEVICES=0
   export PYTHONPATH=.
   ```

2. **Verify Ollama is running**:
   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

3. **Install Ollama models** (if not present):
   ```bash
   ollama pull exaone3.5:latest      # Question generator
   ollama pull deepseek-r1:latest    # Teacher (primary)
   ollama pull qwen2.5:14b          # Teacher (alternative)
   ```

### Phase 1: Generate Questions

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python knowledge3d/training/rlwhf/generate_questions_ollama.py \
  --ollama http://127.0.0.1:11434 \
  --model exaone3.5:latest \
  --target 10000 \
  --max-pdfs 328 \
  --use-wordnet \
  --output /K3D/Knowledge3D.local/datasets/rlwhf/questions_generated.jsonl
```

**Expected time**: 2-4 hours (10K questions, ~1-2 sec per question)

**Validate output**:
```bash
python scripts/validate_generated_questions.py /K3D/Knowledge3D.local/datasets/rlwhf/questions_generated.jsonl
```

### Phase 2: Student Attempts

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python knowledge3d/training/rlwhf/student_attempt_trm.py \
  --questions /K3D/Knowledge3D.local/datasets/rlwhf/questions_generated.jsonl \
  --rpn-embeddings /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
  --trm-weights /K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz \
  --output /K3D/Knowledge3D.local/datasets/rlwhf/student_attempts.jsonl
```

**Expected time**: 10-15 minutes (10K questions)

### Phase 3: Teacher Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python knowledge3d/training/rlwhf/teacher_eval_ollama.py \
  --input /K3D/Knowledge3D.local/datasets/rlwhf/student_attempts.jsonl \
  --ollama http://127.0.0.1:11434 \
  --model deepseek-r1:latest \
  --output /K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl
```

**Expected time**: 3-6 hours (10K questions, ~1-2 sec per evaluation)

**Optimization**: Run in parallel with multiple Ollama instances if available.

### Phases 4-6: Coming Next

After teacher evaluation completes:
- Phase 4: Harvest thinking tags → embeddings
- Phase 5: Build training dataset (questions + targets + thinking + rewards)
- Phase 6: Train TRM with reward-weighted loss

---

## Deliverables

After Phase 1-3 completion, you should have:

1. **Generated Questions**: `questions_generated.jsonl` (10K questions from K3D knowledge)
2. **Student Attempts**: `student_attempts.jsonl` (TRM baseline performance)
3. **Teacher Evaluations**: `teacher_evaluations.jsonl` (ratings + thinking tags + feedback)

**Dataset Statistics to Report**:
- Question difficulty distribution (easy/medium/hard)
- Student convergence rate
- Teacher ratings distribution (good/partial/bad/dishonest)
- Average honesty score from thinking tags
- Average reasoning depth

---

## Error Handling

### If Ollama fails:
- Check Ollama is running: `curl http://127.0.0.1:11434/api/tags`
- Check model is loaded: `ollama list`
- Increase timeout if model is slow
- Retry failed samples (scripts save progress incrementally)

### If questions are low quality:
- Adjust `QUESTION_GENERATOR_SYSTEM_PROMPT` temperature
- Filter by minimum context length
- Sample more diverse PDF sections

### If teacher evaluations are inconsistent:
- Try different teacher model (qwen2.5:14b vs deepseek-r1)
- Lower temperature (more consistent) vs higher (more creative)
- Validate thinking tag parsing is working correctly

---

## Testing

Before running full pipeline, test with small samples:

```bash
# Test Phase 1: Generate 100 questions
python knowledge3d/training/rlwhf/generate_questions_ollama.py --target 100 --output /tmp/test_questions.jsonl

# Test Phase 2: Process 100 questions
python knowledge3d/training/rlwhf/student_attempt_trm.py --questions /tmp/test_questions.jsonl --output /tmp/test_attempts.jsonl

# Test Phase 3: Evaluate 10 samples
python knowledge3d/training/rlwhf/teacher_eval_ollama.py --input /tmp/test_attempts.jsonl --max-samples 10 --output /tmp/test_eval.jsonl
```

---

## Success Criteria

**Phase 1 Success**:
- 10K questions generated
- 70%+ from PDFs, 30% from WordNet
- Difficulty distribution: ~40% easy, 40% medium, 20% hard
- <5% duplicate questions

**Phase 2 Success**:
- 10K student attempts completed
- Convergence rate: 80-100%
- Average output norm: 50-500 (varies by question)

**Phase 3 Success**:
- 10K teacher evaluations completed
- Ratings distribution: ~30-40% good, ~30-40% partial, ~20-30% bad, <5% dishonest
- Thinking tags present in >90% of responses
- Average honesty score: 0.5-0.8
- Average reasoning depth: 3-8

---

**Ready to execute**: All scripts are complete and ready for implementation.
**Estimated total time**: 5-10 hours (mostly Ollama inference)
**Next after completion**: Phases 4-6 (thinking harvesting + dataset building + TRM training)
