from __future__ import annotations

"""
Phase 7 — Chat history consolidation -> Book in Library.

MVP: read recent chat_message embeddings (32-d) from House Memory Chat Book
for the current channel, summarize via average, and add a 'book' object in
the 'Books' room with extra.embedding32. Export updated memory_house.gltf.

CLI:
  python -m knowledge3d.tools.phase7.consolidation \
    --title "Our Quantum Physics Discussion" \
    --channel "#general" \
    --last 100 \
    --out viewer/public/memory_house.gltf
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _avg(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    d = len(vectors[0])
    acc = [0.0] * d
    n = 0
    for v in vectors:
        if not isinstance(v, list) or len(v) != d:
            continue
        for i in range(d):
            acc[i] += float(v[i])
        n += 1
    if n == 0:
        return []
    return [x / n for x in acc]


class ChatHistoryConsolidator:
    def __init__(self) -> None:
        from ..house_memory import MemoryHouse  # type: ignore
        self._MemoryHouse = MemoryHouse

    def _chat_messages(self, house, book_label: str, last: int) -> List[dict]:
        objs = [o for o in house.objects if o.kind == 'chat_message' and (o.extra or {}).get('parent') == house.ensure_chat_book(book_label)]
        # sort by ts
        def _ts(o):
            try:
                from datetime import datetime as _dt
                return _dt.fromisoformat((o.extra or {}).get('ts','').replace('Z','')).timestamp()
            except Exception:
                return 0.0
        objs.sort(key=_ts)
        return objs[-max(1, last):]

    def consolidate_chat_to_book(self, title: str, channel: Optional[str] = None, last: int = 100, out_path: Optional[Path] = None) -> Optional[Path]:
        house = self._MemoryHouse()
        book_label = f"Chat {channel}" if channel else "Chat #general"
        msgs = self._chat_messages(house, book_label, last)
        if not msgs:
            return None
        # Collect embeddings (32-d) — fallback to hashed via export if missing
        vecs: List[List[float]] = []
        for o in msgs:
            e32 = (o.extra or {}).get('embedding32') if isinstance(o.extra, dict) else None
            if isinstance(e32, list) and len(e32) == 32:
                vecs.append([float(x) for x in e32])
        if not vecs:
            return None
        emb = _avg(vecs)
        # Add Library book with explicit embedding
        house.add_room('Books', 'Long-term knowledge books')
        extra = {'embedding32': emb, 'source': 'chat_history', 'author': 'AI Self', 'created_at': datetime.utcnow().isoformat() + 'Z'}
        house.add_object('Books', title[:80], text='consolidated from chat', kind='book', extra=extra)
        # repo root = parents[3] (knowledge3d/tools/phase7 -> knowledge3d -> repo)
        out = out_path or (Path(__file__).resolve().parents[3] / 'viewer' / 'public' / 'memory_house.gltf')
        house.export_gltf(out)
        return out


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', required=True)
    ap.add_argument('--channel', default='#general')
    ap.add_argument('--last', type=int, default=100)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[3] / 'viewer' / 'public' / 'memory_house.gltf'))
    args = ap.parse_args()
    c = ChatHistoryConsolidator()
    path = c.consolidate_chat_to_book(args.title, channel=args.channel, last=int(args.last), out_path=Path(args.out))
    if path is None:
        print('No chat history to consolidate')
    else:
        print(f'Consolidated chat to book — wrote {path}')


if __name__ == '__main__':  # pragma: no cover
    main()
