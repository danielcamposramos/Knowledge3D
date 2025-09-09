from __future__ import annotations

"""
Spatial Text Skill — memory-native answering without external LLMs.

Given a question and (label, text) contexts from the House, compose a
concise answer by selecting diverse snippets and stitching them into a
coherent paragraph. This is a retrieval + composition baseline that
keeps inference fully inside K3D.
"""

from typing import List, Tuple
from datetime import datetime
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
                mode, out = "compose_generate", compose_generate(question, contexts, max_tokens=max_tokens)
                _log_mode_decision(question, ctx_txts, mode, out)
                return mode, out
            else:
                mode, out = "compose", compose_answer(question, contexts, max_chars=max_tokens*3)
                _log_mode_decision(question, ctx_txts, mode, out)
                return mode, out
        except Exception:
            pass
    # Heuristic fallback
    if contexts:
        mode, out = "compose_generate", compose_generate(question, contexts, max_tokens=max_tokens)
        _log_mode_decision(question, ctx_txts, mode, out)
        return mode, out
    mode, out = "compose", compose_answer(question, contexts, max_chars=max_tokens*3)
    _log_mode_decision(question, ctx_txts, mode, out)
    return mode, out


def _log_mode_decision(question: str, ctx_txts: List[str], mode: str, answer: str) -> None:
    """Append a JSONL record for mode selection outcomes.

    Writes to docs/reports/training/mode_selector_outcomes.jsonl and mirrors to
    ../Knowledge3D.local/logs/mode_selector-<date>.jsonl. If K3D_MODE_LOG_SIM=1
    and sentence-transformers is available, compute a quick similarity score
    between the answer and context blob for downstream labels.
    """
    try:
        import os, json
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        out_repo = repo_root / "docs" / "reports" / "training" / "mode_selector_outcomes.jsonl"
        local_root = repo_root.parent / f"{repo_root.name}.local"
        ts = datetime.utcnow().isoformat() + "Z"
        rec = {"ts": ts, "question": question, "mode": mode, "answer": answer, "contexts": ctx_txts[:6]}
        # Optional similarity for outcome label
        want_sim = os.getenv("K3D_MODE_LOG_SIM", "0").strip() != "0"
        if want_sim:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                import numpy as _np  # type: ignore
                import torch  # type: ignore
                st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=("cuda" if torch.cuda.is_available() else "cpu"))
                e1 = st.encode([answer], convert_to_numpy=True)[0]
                blob = "\n".join(ctx_txts[:4])
                e2 = st.encode([blob], convert_to_numpy=True)[0]
                sim = float(_np.dot(e1, e2) / (float(_np.linalg.norm(e1)) * float(_np.linalg.norm(e2)) + 1e-9))
                rec["sim"] = sim
            except Exception:
                pass
        out_repo.parent.mkdir(parents=True, exist_ok=True)
        with out_repo.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # mirror to local logs
        try:
            local_root.mkdir(parents=True, exist_ok=True)
            day = ts[:10]
            out_local = local_root / "logs" / f"mode_selector-{day}.jsonl"
            out_local.parent.mkdir(parents=True, exist_ok=True)
            with out_local.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
    except Exception:
        pass
