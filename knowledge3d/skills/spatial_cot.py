from __future__ import annotations

"""
Spatial Chain-of-Thought (CoT) helpers for K3D.

Phase 1: Build a minimal ReasoningPath from a question and House contexts
and provide a payload suitable for viewer visualization. This wrapper
leverages existing compose_* functions without changing their return
types, to keep compatibility with current call sites.

A ReasoningPath is a sequence of steps with operations
(retrieve/compare/synthesize/verify), confidence scores, and citations.
The viewer overlays the resulting path across 3D waypoints corresponding
to memory nodes (resolved by the live server using the dataset graph).
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class ReasoningStep:
    op: str  # retrieve | compare | synthesize | verify
    label: str
    confidence: float
    citation: Optional[str] = None
    verified: Optional[bool] = None
    id: Optional[str] = None  # optional id; live server can resolve by label


@dataclass
class ReasoningPath:
    question: str
    mode: str  # compose | compose_generate | compose_auto
    steps: List[ReasoningStep] = field(default_factory=list)

    def to_payload(self) -> Dict:
        return {
            "question": self.question,
            "mode": self.mode,
            "steps": [asdict(s) for s in self.steps],
        }


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().split())


def _bow(s: str) -> set:
    return {w for w in s.lower().split() if w and (w.isalpha() or any(c.isalnum() for c in w))}


def _mmr_select(question: str, contexts: List[Tuple[str, str]], k: int = 4) -> List[Tuple[str, str]]:
    """Lightweight selection akin to spatial_text.compose_answer.

    Returns up to k (label, text) pairs chosen with a simple MMR flavor
    using bag-of-words overlap. This is duplicated intentionally to avoid
    changing spatial_text's return signature while keeping selection
    consistent for COT tracing.
    """
    q = _normalize(question)
    # Deduplicate labels and normalize
    seen = set(); pairs: List[Tuple[str, str]] = []
    for lab, txt in contexts:
        lab = _normalize(lab); txt = _normalize(txt)
        if not lab or not txt: continue
        if lab in seen: continue
        seen.add(lab); pairs.append((lab, txt))
    if not pairs:
        return []
    # Score relevance by overlap
    qv = _bow(q)
    scored = []
    for lab, txt in pairs:
        tv = _bow(txt)
        rel = len(qv & tv) / (1.0 + len(tv))
        scored.append((rel, lab, txt))
    scored.sort(reverse=True)
    chosen: List[Tuple[str, str]] = []
    for rel, lab, txt in scored:
        if not chosen:
            chosen.append((lab, txt)); continue
        tv = _bow(txt)
        dup = False
        for _, tprev in chosen:
            if len(_bow(tprev) & tv) / (1.0 + len(tv)) > 0.6:
                dup = True; break
        if not dup:
            chosen.append((lab, txt))
        if len(chosen) >= max(1, k):
            break
    return chosen


def _confidence_from_overlap(question: str, text: str) -> float:
    qv = _bow(question)
    tv = _bow(text)
    if not tv:
        return 0.0
    return max(0.0, min(1.0, len(qv & tv) / max(1, len(qv))))


def _verify_step(question: str, label: str, text: str) -> Tuple[bool, float]:
    """Basic verification: ensure at least one question token appears in the citation text.

    Returns (ok, score) where score is the same confidence heuristic used
    for retrieve steps.
    """
    conf = _confidence_from_overlap(question, text)
    ok = conf >= 0.2  # conservative threshold for Phase 1
    return ok, conf


def compose_with_cot(
    question: str,
    contexts: List[Tuple[str, str]],
    mode: str = "compose",  # compose | compose_generate | compose_auto
    composer: Optional[Callable[[str, List[Tuple[str, str]]], str]] = None,
) -> Tuple[str, ReasoningPath]:
    """Compose an answer using existing composer while building a ReasoningPath.

    - mode selects which composer intent is used (informational only)
    - composer can be provided; otherwise, a best-effort local import of
      spatial_text.compose_answer/generate is performed according to mode.
    - Returns (text, ReasoningPath)
    """
    q = _normalize(question)
    rp = ReasoningPath(question=q, mode=mode, steps=[])

    # Selection and retrieve steps
    selected = _mmr_select(q, contexts, k=4)
    for lab, txt in selected:
        conf = _confidence_from_overlap(q, txt)
        rp.steps.append(ReasoningStep(op="retrieve", label=lab, confidence=float(conf), citation=txt))

    # Compare steps: pairwise adjacent comparisons (lightweight)
    for i in range(max(0, len(selected) - 1)):
        a_lab, a_txt = selected[i]
        b_lab, b_txt = selected[i + 1]
        # Compare by overlap between citations to encourage diversity
        ov = len(_bow(a_txt) & _bow(b_txt)) / (1.0 + len(_bow(a_txt) | _bow(b_txt)))
        conf = max(0.0, min(1.0, 1.0 - ov))  # higher when diverse
        rp.steps.append(ReasoningStep(op="compare", label=f"{a_lab} ↔ {b_lab}", confidence=float(conf), citation=None))

    # Synthesize step: presence indicates composition completed
    if selected:
        labs = ", ".join(lab for lab, _ in selected[:3])
        rp.steps.append(ReasoningStep(op="synthesize", label=f"{labs}", confidence=0.7, citation=None))

    # Verify each retrieved citation
    for lab, txt in selected:
        ok, conf = _verify_step(q, lab, txt)
        rp.steps.append(ReasoningStep(op="verify", label=lab, confidence=float(conf), citation=txt, verified=bool(ok)))

    # Perform composition using provided or inferred composer
    text: str = ""
    if composer is not None:
        text = composer(q, contexts)
    else:
        try:
            from .spatial_text import compose_answer, compose_generate  # type: ignore
            if mode == "compose_generate":
                text = compose_generate(q, contexts, max_tokens=256)
            else:
                text = compose_answer(q, contexts)
        except Exception:
            # Fallback text when composer import fails
            lines = [f"- {lab}: {txt[:160]}" for lab, txt in selected[:4]]
            text = ("I will answer from nearby memory:\n" + "\n".join(lines)) if lines else (
                "I don't have enough memory text for that yet.")

    return text, rp

