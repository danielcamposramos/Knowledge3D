from __future__ import annotations

from pathlib import Path
from typing import List
import re


class DatasetLocator:
    def __init__(self, repo_path: str):
        self.repo = Path(repo_path)

    def find_dataset_sources(self) -> List[str]:
        sources: List[str] = []
        datasets_md = self.repo / 'DATASETS.md'
        if datasets_md.exists():
            sources.extend(self.parse_markdown_datasets(datasets_md))
        datasets_dir = self.repo / 'datasets'
        if datasets_dir.exists():
            sources.extend(self.parse_directory_datasets(datasets_dir))
        readme = self.repo / 'README.md'
        if readme.exists():
            sources.extend(self.parse_readme_datasets(readme))
        # normalize and dedupe
        out: List[str] = []
        seen = set()
        for s in sources:
            if not isinstance(s, str):
                continue
            u = s.strip()
            if not u or not u.startswith(('http://','https://')):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def parse_markdown_datasets(self, filepath: Path) -> List[str]:
        urls: List[str] = []
        txt = filepath.read_text(encoding='utf-8')
        urls.extend(re.findall(r'https?://[^\s\)]+', txt))
        return urls

    def parse_directory_datasets(self, dirpath: Path) -> List[str]:
        urls: List[str] = []
        for fp in dirpath.rglob('*'):
            if fp.name.lower() in {'source.txt','sources.txt','readme.md'}:
                try:
                    txt = fp.read_text(encoding='utf-8')
                    urls.extend(re.findall(r'https?://[^\s\)]+', txt))
                except Exception:
                    continue
        return urls

    def parse_readme_datasets(self, filepath: Path) -> List[str]:
        urls: List[str] = []
        ok = False
        for ln in filepath.read_text(encoding='utf-8').splitlines():
            if re.search(r'^##\s+Datasets', ln, flags=re.I):
                ok = True
                continue
            if ok and ln.startswith('#'):
                break
            if ok:
                urls.extend(re.findall(r'https?://[^\s\)]+', ln))
        return urls


def main():  # pragma: no cover
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    args = ap.parse_args()
    loc = DatasetLocator(args.repo)
    srcs = loc.find_dataset_sources()
    print(json.dumps(srcs, indent=2))


if __name__ == '__main__':  # pragma: no cover
    main()

