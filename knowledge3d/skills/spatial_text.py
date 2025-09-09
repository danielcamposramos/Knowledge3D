from __future__ import annotations

"""
Spatial Text Skill — memory-native answering without external LLMs.

Given a question and (label, text) contexts from the House, compose a
concise answer by selecting diverse snippets and stitching them into a
coherent paragraph. This is a retrieval + composition baseline that
keeps inference fully inside K3D.
"""

from typing import List, Tuple
from pathlib import Path
from .llm import LLMSkill, LLMConfig  # type: ignore

_ranker = None
_selector = None

def _load_ranker():
    global _ranker
    if _ranker is not None:
        return _ranker
    try:
        from pathlib import Path
        from ..models.answer_ranker import load  # type: ignore
        repo_root = Path(__file__).resolve().parents[2]
        model_path = (repo_root.parent / f"{repo_root.name}.local" / "models" / "answer_ranker.pkl")
        if model_path.exists():
            _ranker = load(model_path)
    except Exception:
        _ranker = None
    return _ranker


def _load_selector():
    global _selector
    if _selector is not None:
        return _selector
    try:
        from ..models.mode_selector import load as _load  # type: ignore
        repo_root = Path(__file__).resolve().parents[2]
        model_path = (repo_root.parent / f"{repo_root.name}.local" / "models" / "mode_selector.pkl")
        if model_path.exists():
            _selector = _load(model_path)
    except Exception:
        _selector = None
    return _selector


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().split())


def compose_answer(question: str, contexts: List[Tuple[str, str]], max_chars: int = 600) -> str:
    q = _normalize(question)
    # Optional: learned ranker to reorder contexts by predicted reward
    rk = _load_ranker()
    if rk is not None:
        try:
            contexts = sorted(contexts, key=lambda p: rk.score(q, p[1] or ''), reverse=True)
        except Exception:
            pass
    # Deduplicate by label, keep first non-empty text
    seen = set()
    pairs: List[Tuple[str, str]] = []
    for lab, txt in contexts:
        lab = _normalize(lab)
        txt = _normalize(txt)
        if not lab or not txt:
            continue
        if lab in seen:
            continue
        seen.add(lab)
        pairs.append((lab, txt))
        if len(pairs) >= 8:
            break
    if not pairs:
        return "I don't have enough memory text for that yet. Try exploring a few nodes first."
    # Simple max-marginal-relevance (MMR) selection on bag-of-words overlap
    def bow(s: str) -> set:
        return {w for w in s.lower().split() if w.isalpha() or any(c.isalnum() for c in w)}
    qv = bow(q)
    scored = []
    for lab, txt in pairs:
        tv = bow(txt)
        rel = len(qv & tv) / (1.0 + len(tv))
        scored.append((rel, lab, txt))
    scored.sort(reverse=True)
    chosen: List[Tuple[str, str]] = []
    for rel, lab, txt in scored:
        if not chosen:
            chosen.append((lab, txt)); continue
        dup = False
        tv = bow(txt)
        for _, tprev in chosen:
            if len(bow(tprev) & tv) / (1.0 + len(tv)) > 0.6:
                dup = True; break
        if not dup:
            chosen.append((lab, txt))
        if len(chosen) >= 4:
            break
    # Compose narrative
    parts = [f"Question: {q}"]
    parts.append("From my memory:")
    budget = max_chars
    out_lines: List[str] = []
    for lab, txt in chosen:
        line = f"- {lab}: {txt}"
        if len("\n".join(out_lines) + line) > budget:
            break
        out_lines.append(line)
    parts.extend(out_lines)
    parts.append("I hope this helps. We can navigate to any of these topics for more detail.")
    return "\n".join(parts)


def compose_generate(question: str, contexts: List[Tuple[str, str]], max_tokens: int = 256) -> str:
    """
    Grounded generative path: uses the internal LLM skill to generate an answer
    strictly from provided contexts (House memory). Honors the grounded policy
    in knowledge3d/skills/llm.py by instructing the model to admit unknowns and
    cite labels.

    Model selection follows env defaults, so RLWHF LoRA adapters can be used by
    setting `K3D_LLM_MODEL` to the adapter directory and `K3D_LLM_PEFT_BASE` to
    the base model id.
    """
    q = _normalize(question)
    # Limit to unique, non-empty contexts
    seen = set(); pairs: List[Tuple[str, str]] = []
    for lab, txt in contexts:
        lab = _normalize(lab); txt = _normalize(txt)
        if not lab or not txt: continue
        if lab in seen: continue
        seen.add(lab); pairs.append((lab, txt))
        if len(pairs) >= 6: break
    if not pairs:
        return "I don't have enough memory text for that yet."
    llm = LLMSkill(LLMConfig())
    return llm.answer_with_rag(q, pairs, max_tokens=max_tokens)


def compose_auto(question: str, contexts: List[Tuple[str, str]], max_tokens: int = 256) -> Tuple[str, str]:
    """
    Auto-select between compose (retrieval+stitching) and compose_generate
    (grounded generative) using the trained mode selector when available.

    Returns (mode, text) where mode is "compose" or "compose_generate".
    """
    sel = _load_selector()
    # Prepare contexts for selector (only text fields)
    ctx_txts = [ (txt or "") for _, txt in contexts ]
    if sel is not None:
        try:
            y = sel.predict(question or "", ctx_txts)
            if y == 1:
                return "compose_generate", compose_generate(question, contexts, max_tokens=max_tokens)
            else:
                return "compose", compose_answer(question, contexts, max_chars=max_tokens*3)
        except Exception:
            pass
    # Heuristic fallback
    if contexts:
        return "compose_generate", compose_generate(question, contexts, max_tokens=max_tokens)
    return "compose", compose_answer(question, contexts, max_chars=max_tokens*3)
