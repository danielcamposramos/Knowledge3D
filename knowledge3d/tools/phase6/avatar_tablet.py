from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import hashlib
import json


def _hash_vec(text: str, dims: int = 32) -> List[float]:
    h = hashlib.sha256(text.encode('utf-8')).digest()
    vals: List[float] = []
    i = 0
    while len(vals) < dims:
        b = h[i % len(h)]
        vals.append((b / 255.0) - 0.5)
        i += 1
    return vals


@dataclass
class App:
    name: str
    embedding: List[float]
    version: str
    installed_at: str


class MockScreen:
    def __init__(self, screen_id: str, out_dir: Optional[Path] = None) -> None:
        self.screen_id = screen_id
        repo = Path(__file__).resolve().parents[2]
        self.out_dir = out_dir or (repo / 'viewer' / 'public' / 'projections')
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def display_content(self, embedding: List[float]) -> Path:
        safe = ''.join(c for c in self.screen_id if c.isalnum() or c in ('-','_')).strip() or 'screen'
        path = self.out_dir / f'{safe}.txt'
        summary = {
            'screen_id': self.screen_id,
            'ts': datetime.utcnow().isoformat() + 'Z',
            'embedding_dims': len(embedding),
            'embedding_head': [float(x) for x in embedding[:8]],
        }
        path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return path


class AvatarTablet:
    def __init__(self) -> None:
        self.installed_apps: List[App] = []
        self.current_app: Optional[App] = None

    def install_app(self, app_name: str, app_embedding: Optional[List[float]] = None) -> bool:
        emb = list(app_embedding) if isinstance(app_embedding, list) else _hash_vec(app_name, 64)
        app = App(name=app_name, embedding=emb, version='1.0', installed_at=datetime.utcnow().isoformat() + 'Z')
        # replace if exists
        self.installed_apps = [a for a in self.installed_apps if a.name != app_name]
        self.installed_apps.append(app)
        self.current_app = app
        return True

    def find_app(self, app_name: str) -> Optional[App]:
        for a in self.installed_apps:
            if a.name == app_name:
                return a
        return None

    def find_screen(self, screen_id: str) -> MockScreen:
        return MockScreen(screen_id)

    def cast_to_screen(self, app_name: str, screen_id: str) -> Optional[Path]:
        app = self.find_app(app_name)
        if not app:
            return None
        screen = self.find_screen(screen_id)
        return screen.display_content(app.embedding)

