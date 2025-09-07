from __future__ import annotations

"""
Spatial Text Skill — memory-native answering without external LLMs.

Given a question and (label, text) contexts from the House, compose a
concise answer by selecting diverse snippets and stitching them into a
coherent paragraph. This is a retrieval + composition baseline that
keeps inference fully inside K3D.
"""

from typing import List, Tuple


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().split())


def compose_answer(question: str, contexts: List[Tuple[str, str]], max_chars: int = 600) -> str:
    q = _normalize(question)
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

