import math
from typing import Dict, Tuple

from .knowledge_sectors import get_sector_angle, _load as load_sectors


def tree_complexity(tree: dict) -> float:
    # Complexity = nodes + depth weight
    def walk(n: dict, d=1):
        cnt = 1
        md = d
        for ch in n.get('children', []) or []:
            c, dm = walk(ch, d+1)
            cnt += c
            md = max(md, dm)
        return cnt, md
    nodes, depth = walk(tree)
    return float(nodes + depth * 2)


def calculate_tree_position(domain: str, tree: dict, max_radius: float = 10.0) -> tuple[float, float, float, float]:
    angle_deg = get_sector_angle(domain)
    ang = angle_deg * math.pi / 180.0
    comp = tree_complexity(tree)
    dist = min(comp * 0.2, max_radius)
    x = dist * math.cos(ang)
    z = dist * math.sin(ang)
    y = 0.0
    rot = (ang + math.pi)  # face center
    return float(x), float(y), float(z), float(rot)


def current_sectors() -> Dict[str, Tuple[float, float]]:
    return load_sectors()

