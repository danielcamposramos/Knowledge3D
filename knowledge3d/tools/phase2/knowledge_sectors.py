import json
from dataclasses import dataclass
from math import fmod
from pathlib import Path
from typing import Dict, Tuple


GARDEN_DIR = Path('viewer/public/knowledge_garden')
SECTORS_PATH = GARDEN_DIR / 'knowledge_sectors.json'


DEFAULT_SECTORS: Dict[str, Tuple[float, float]] = {
    'Physics': (0.0, 60.0),
    'Biology': (60.0, 120.0),
    'Mathematics': (120.0, 180.0),
    'Philosophy': (180.0, 240.0),
    'Art': (240.0, 300.0),
    'Engineering': (300.0, 360.0),
}


def _load() -> Dict[str, Tuple[float, float]]:
    # Migrate legacy path if exists
    legacy = Path('viewer/public/knowledge_sectors.json')
    if SECTORS_PATH.exists() or legacy.exists():
        try:
            if SECTORS_PATH.exists():
                data = json.loads(SECTORS_PATH.read_text(encoding='utf-8'))
            else:
                data = json.loads(legacy.read_text(encoding='utf-8'))
            out: Dict[str, Tuple[float, float]] = {}
            for k, v in data.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    out[k] = (float(v[0]), float(v[1]))
            if out:
                return out
        except Exception:
            pass
    return dict(DEFAULT_SECTORS)


def _save(sectors: Dict[str, Tuple[float, float]]) -> None:
    SECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {k: [float(a), float(b)] for k, (a, b) in sectors.items()}
    SECTORS_PATH.write_text(json.dumps(data, indent=2), encoding='utf-8')


def get_sector_angle(domain: str) -> float:
    """Return central angle for a domain, allocating a new sector if needed.

    Tries fuzzy matching: if an existing sector name is a substring of the domain
    (case-insensitive), use that sector.
    """
    name = (domain or '').strip()
    sectors = _load()
    # direct match
    if name in sectors:
        a, b = sectors[name]
        return (a + b) * 0.5
    # fuzzy match
    lname = name.lower()
    for key in sectors.keys():
        if key.lower() in lname or lname in key.lower():
            a, b = sectors[key]
            return (a + b) * 0.5
    # allocate new
    angle = allocate_new_sector(name, sectors)
    return angle


def allocate_new_sector(domain: str, sectors: Dict[str, Tuple[float, float]] | None = None) -> float:
    """Insert a new sector into the largest angular gap and persist.

    Returns central angle degrees of the new sector.
    """
    if sectors is None:
        sectors = _load()
    # Collect existing sector boundaries
    bounds = []
    for a, b in sectors.values():
        a = float(a) % 360.0; b = float(b) % 360.0
        bounds.append(a); bounds.append(b)
    bounds = sorted(set(bounds)) or [0.0]
    # Find largest gap on the circle
    max_gap = -1.0
    insert_angle = 0.0
    for i in range(len(bounds)):
        a = bounds[i]
        b = bounds[(i + 1) % len(bounds)]
        gap = (b - a) if b > a else (360.0 - a + b)
        if gap > max_gap:
            max_gap = gap
            insert_angle = (a + gap * 0.5) % 360.0
    # Allocate a 30-degree slice centered at insert_angle
    half = 15.0
    new_start = (insert_angle - half) % 360.0
    new_end = (insert_angle + half) % 360.0
    sectors[domain] = (new_start, new_end)
    _save(sectors)
    return insert_angle


if __name__ == '__main__':
    # basic smoke test
    print('Angle Physics:', get_sector_angle('Physics'))
    print('Angle Quantum Physics:', get_sector_angle('Quantum Physics'))
    print('Angle NewDomainX:', get_sector_angle('NewDomainX'))
