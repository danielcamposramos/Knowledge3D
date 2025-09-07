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


def _read_jsonl(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    caps: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                cap = str(j.get("caption") or "").strip()
                cid = j.get("id")
                if not cid:
                    # derive id from url or caption hash
                    u = j.get("url") or ""
                    raw = (str(u) + cap).encode("utf-8")
                    cid = md5(raw).hexdigest()[:16]
                cid = str(cid)
                if cap:
                    ids.append(cid)
                    caps.append(cap)
            except Exception:
                continue
    return ids, caps


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Match audio and video by caption similarity")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30000, help="number of top pairs to output")
    args = ap.parse_args()

    a_ids, a_caps = _read_jsonl(Path(args.audio))
    v_ids, v_caps = _read_jsonl(Path(args.video))
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
