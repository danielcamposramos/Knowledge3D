from __future__ import annotations

"""
Match audio and video items via caption similarity (TF‑IDF) to prioritize
cross‑modal connectivity in the 50k selection.

Inputs
- audio_meta.jsonl: lines of {id, path, caption}
- video_meta.jsonl: lines of {id, path, caption}

Outputs
- matches.jsonl: lines of {audio_id, video_id, score}
- pool.txt: union of captions from matched pairs (for image/text selection)

Usage
  python -m knowledge3d.tools.match_crossmodal \
    --audio ../Knowledge3D.local/datasets/audiocaps/meta.jsonl \
    --video ../Knowledge3D.local/datasets/msrvtt/meta.jsonl \
    --out ../Knowledge3D.local/datasets/matched
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


from hashlib import md5


def _read_any_meta(path: Path) -> Tuple[List[str], List[str]]:
    """Read either JSONL (one object per line) or a JSON list.

    Accepts objects with keys: id (optional), caption (preferred) or text.
    If id is missing, derives a stable id from url+caption or caption hash.
    """
    ids: List[str] = []
    caps: List[str] = []
    txt = path.read_text(encoding="utf-8")
    # Heuristic: JSONL won't start with '['
    if not txt.lstrip().startswith("["):
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            cap = str(j.get("caption") or j.get("text") or "").strip()
            if not cap:
                continue
            cid = j.get("id")
            if not cid:
                u = j.get("url") or j.get("image") or j.get("video") or ""
                raw = (str(u) + cap).encode("utf-8")
                cid = md5(raw).hexdigest()[:16]
            ids.append(str(cid))
            caps.append(cap)
        return ids, caps
    # JSON array
    try:
        arr = json.loads(txt)
    except Exception:
        return ids, caps
    if not isinstance(arr, list):
        return ids, caps
    for j in arr:
        try:
            cap = str((j.get("caption") if isinstance(j, dict) else "") or (j.get("text") if isinstance(j, dict) else "") or "").strip()
        except Exception:
            continue
        if not cap:
            continue
        cid = None
        if isinstance(j, dict):
            cid = j.get("id")
        if not cid:
            u = (j.get("url") if isinstance(j, dict) else None) or (j.get("image") if isinstance(j, dict) else None) or (j.get("video") if isinstance(j, dict) else None) or ""
            raw = (str(u) + cap).encode("utf-8")
            cid = md5(raw).hexdigest()[:16]
        ids.append(str(cid))
        caps.append(cap)
    return ids, caps


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Match audio and video by caption similarity")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30000, help="number of top pairs to output")
    args = ap.parse_args()

    a_ids, a_caps = _read_any_meta(Path(args.audio))
    v_ids, v_caps = _read_any_meta(Path(args.video))
    if not a_ids or not v_ids:
        print("Empty inputs; nothing to match")
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        raise SystemExit("Please install scikit-learn and numpy") from e

    vec = TfidfVectorizer(max_features=100000)
    A = vec.fit_transform(a_caps)
    V = vec.transform(v_caps)
    # Chunked similarity: compute in slices to avoid RAM spikes
    top_k = int(args.top)
    pairs: List[Tuple[float, int, int]] = []
    step = max(1, len(a_ids) // 20)
    for i0 in range(0, len(a_ids), step):
        i1 = min(len(a_ids), i0 + step)
        S = (A[i0:i1] @ V.T).toarray()
        # For each audio row, take best video match
        for r, row in enumerate(S):
            j = int(row.argmax())
            s = float(row[j])
            pairs.append((s, i0 + r, j))
    # Take top pairs globally
    pairs.sort(reverse=True, key=lambda t: t[0])
    pairs = pairs[:top_k]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "matches.jsonl").open("w", encoding="utf-8") as f:
        for s, ai, vj in pairs:
            f.write(json.dumps({"audio_id": a_ids[ai], "video_id": v_ids[vj], "score": s}) + "\n")
    # Write caption pool
    caps: List[str] = []
    seen = set()
    for _, ai, vj in pairs:
        for c in (a_caps[ai], v_caps[vj]):
            if c not in seen:
                seen.add(c)
                caps.append(c)
    (out_dir / "pool.txt").write_text("\n".join(caps), encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs and {len(caps)} pooled captions to {out_dir}")


if __name__ == "__main__":
    main()
