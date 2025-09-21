from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import requests  # type: ignore


OLLAMA_HOST = "http://192.168.0.4:11434"


class BookProcessor:
    def __init__(
        self,
        book_dir: str,
        ollama_model: str = "exaone3.5:latest",
        max_chars: int = 8000,
        per_book_timeout: int = 90,
        first_call_timeout: int = 180,
        system_prompt: Optional[str] = None,
    ):
        self.book_dir = Path(book_dir)
        self.ollama_model = ollama_model
        self.max_chars = int(max_chars)
        self.per_book_timeout = int(per_book_timeout)
        self.first_call_timeout = int(first_call_timeout)
        self.system_prompt = system_prompt
        self._first_generate = True

    def process_books(self, limit: Optional[int] = None) -> List[Dict]:
        """Process JSON books — distill thinking tags using Ollama over HTTP.

        - Uses one model per usage (keep_alive='0s')
        - Truncates content to max_chars to respect context windows
        - Returns list of {book_id, title, thinking_tags, source_path}
        """
        if not self.book_dir.exists():
            print(f"⚠️ Book directory not found: {self.book_dir}")
            return []
        thinking_tags: List[Dict] = []
        count = 0
        for fp in sorted(self.book_dir.glob("*.json")):
            try:
                raw_book = self.load_book(fp)
                if isinstance(raw_book, list):
                    meta: Dict[str, str] = {"title": "", "content": ""}
                    titles = []
                    parts = []
                    for entry in raw_book:
                        if not isinstance(entry, dict):
                            continue
                        if isinstance(entry.get("title"), str):
                            titles.append(str(entry["title"]))
                        text = entry.get("content") or entry.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                    meta["title"] = " ".join(titles).strip()
                    meta["content"] = "\n".join(parts)
                    book = meta
                else:
                    book = raw_book
                tags = self.distill_thinking_tags(book)
                thinking_tags.append({
                    'book_id': fp.stem,
                    'title': book.get('title', ''),
                    'thinking_tags': tags,
                    'source_path': str(fp)
                })
                print(f"✅ Distilled thinking tags from {fp.name} ({len(tags)} tags)")
                count += 1
                if limit is not None and count >= int(limit):
                    break
            except Exception as e:
                print(f"❌ Failed on {fp.name}: {e}")
                continue
        return thinking_tags

    def load_book(self, filepath: Path) -> dict:
        with filepath.open('r', encoding='utf-8') as f:
            return json.load(f)

    def extract_content(self, book: dict) -> str:
        if isinstance(book, dict):
            # Common fields: title, content, chapters
            parts: List[str] = []
            if isinstance(book.get('title'), str):
                parts.append(str(book['title']))
            if isinstance(book.get('content'), str):
                parts.append(str(book['content']))
            if isinstance(book.get('chapters'), list):
                for ch in book['chapters']:
                    if isinstance(ch, dict) and isinstance(ch.get('text'), str):
                        parts.append(str(ch['text']))
            txt = "\n".join(parts)
        else:
            txt = str(book)
        return txt[: self.max_chars]

    def _ollama_generate(self, prompt: str, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """Call Ollama HTTP API with keep_alive=0 to ensure unload after usage."""
        url = f"{OLLAMA_HOST}/api/generate"
        payload = {
            "model": str(model or self.ollama_model),
            "prompt": prompt,
            "stream": False,
            # Unload immediately to keep memory/context clean
            "keep_alive": "0s",
        }
        # Include optional system prompt
        if self.system_prompt:
            payload["system"] = self.system_prompt
        # Longer timeout on first generate to allow model load
        if self._first_generate:
            to = int(self.first_call_timeout)
            self._first_generate = False
        else:
            to = int(timeout or self.per_book_timeout)
        r = requests.post(url, json=payload, timeout=to)
        r.raise_for_status()
        data = r.json()
        return str(data.get("response", "")).strip()

    def distill_thinking_tags(self, book: dict) -> List[str]:
        content = self.extract_content(book)
        if not content:
            return []
        prompt = (
            "Extract concise 'thinking tags' (comma-separated) that capture programmatic thinking patterns in the text.\n"
            "Examples: break into subproblems, test edge cases, use recursion, memoize repeated work, greedy choice, dynamic programming, two-pointer, divide-and-conquer.\n\n"
            f"Text:\n{content}\n\n"
            "Thinking Tags:"
        )
        try:
            out = self._ollama_generate(prompt)
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return []
        # Parse tags
        tags = [t.strip() for t in out.split(',') if t.strip()]
        # De-duplicate and normalize
        uniq: List[str] = []
        seen = set()
        for t in tags:
            k = t.lower()
            if k and k not in seen:
                seen.add(k)
                uniq.append(t)
        return uniq


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--books', default='/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/JSON')
    ap.add_argument('--model', default='exaone3.5:latest')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--output', default=None, help='Optional output JSON path (default viewer/public/books/thinking_tags.json)')
    args = ap.parse_args()
    bp = BookProcessor(args.books, ollama_model=args.model)
    tags = bp.process_books(limit=args.limit)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(tags, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Wrote {out_path} ({len(tags)} entries)")
    else:
        out_dir = Path('viewer/public/books')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'thinking_tags.json'
        out_path.write_text(json.dumps(tags, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Wrote {out_path} ({len(tags)} entries)")


if __name__ == '__main__':  # pragma: no cover
    main()
