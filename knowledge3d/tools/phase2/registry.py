import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .knowledge_sectors import _load as load_sectors


GARDEN_DIR = Path('viewer/public/knowledge_garden')
TREES_DIR = GARDEN_DIR / 'trees'
REGISTRY_PATH = GARDEN_DIR / 'garden_registry.json'
ERROR_LOG = GARDEN_DIR / 'garden_errors.log'


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def ensure_dirs():
    TREES_DIR.mkdir(parents=True, exist_ok=True)


def default_registry() -> Dict[str, Any]:
    return {
        'garden_version': '3.0',
        'center_position': [0.0, 0.0, 0.0],
        'current_radius': 10.0,
        'max_radius': 100.0,
        'trees': [],
        'sectors': load_sectors(),
    }


def load_registry() -> Dict[str, Any]:
    ensure_dirs()
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    reg = default_registry()
    save_registry(reg)
    return reg


def save_registry(reg: Dict[str, Any]) -> None:
    ensure_dirs()
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding='utf-8')


def append_error(msg: str) -> None:
    ensure_dirs()
    ERROR_LOG.write_text((ERROR_LOG.read_text() if ERROR_LOG.exists() else '') + f"[{utc_now()}] {msg}\n", encoding='utf-8')


def register_tree(tree_id: str, filepath: str, domain: str, position: Tuple[float, float, float], rotation: float, sector: str, status: str = 'success', error_code: float | None = None) -> Dict[str, Any]:
    reg = load_registry()
    entry = next((t for t in reg['trees'] if t.get('tree_id') == tree_id), None)
    item = {
        'tree_id': tree_id,
        'filepath': filepath,
        'position': list(map(float, position)),
        'rotation': float(rotation),
        'sector': sector,
        'domain': domain,
        'load_status': status,
        'error_code': error_code,
        'last_loaded': utc_now(),
    }
    if entry:
        entry.update(item)
    else:
        reg['trees'].append(item)
    # Auto-expand current radius if needed
    dist = (position[0] ** 2 + position[2] ** 2) ** 0.5
    if float(reg.get('current_radius', 10.0)) < dist + 1.0:
        reg['current_radius'] = min(float(reg.get('max_radius', 100.0)), dist + 1.0)
    save_registry(reg)
    return reg

