import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

from .knowledge_sectors import _load as load_sectors


def make_svg(sectors: Dict[str, Tuple[float, float]], out_path: Path, radius: float = 200.0, trees: list[dict] | None = None):
    cx, cy = radius + 10, radius + 10
    W = H = int((radius + 10) * 2)
    # Colors
    palette = ["#4cc9f0", "#80ed99", "#e9c46a", "#f4a261", "#e76f51", "#cdb4db", "#90caf9", "#a5d6a7"]
    # Build SVG
    segs = []
    items = list(sectors.items())
    for i, (name, (a0, a1)) in enumerate(items):
        c = palette[i % len(palette)]
        a0r = math.radians(a0); a1r = math.radians(a1)
        x0, y0 = cx + radius * math.cos(a0r), cy + radius * math.sin(a0r)
        x1, y1 = cx + radius * math.cos(a1r), cy + radius * math.sin(a1r)
        # large-arc-flag
        da = (a1 - a0) % 360.0
        laf = 1 if da > 180 else 0
        path = f"M {cx},{cy} L {x0},{y0} A {radius},{radius} 0 {laf},1 {x1},{y1} Z"
        segs.append(f"<path d='{path}' fill='{c}' fill-opacity='0.15' stroke='{c}' stroke-opacity='0.6' stroke-width='1'/>")
        # label at center angle
        ang = math.radians((a0 + a1) * 0.5)
        lx, ly = cx + (radius * 0.65) * math.cos(ang), cy + (radius * 0.65) * math.sin(ang)
        segs.append(f"<text x='{lx:.1f}' y='{ly:.1f}' font-size='12' text-anchor='middle' fill='{c}'>{name}</text>")
    # Trees
    if trees:
        for t in trees:
            px, py, pz = t.get('position', [0.0, 0.0, 0.0])
            x, y = cx + (px / (radius/200.0)), cy + (pz / (radius/200.0))
            segs.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='#00acc1' stroke='#006064' stroke-width='1'/>")
            segs.append(f"<text x='{x+6:.1f}' y='{y-6:.1f}' font-size='10' fill='#00acc1'>{t.get('domain','')}</text>")
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>
<rect width='100%' height='100%' fill='#111'/>
<g>
{''.join(segs)}
</g>
</svg>"""
    out_path.write_text(svg, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--garden', default='viewer/public/knowledge_garden.glb')
    ap.add_argument('--out', default='viewer/public/garden_map.svg')
    args = ap.parse_args()
    sectors = load_sectors()
    trees = []
    gp = Path(args.garden)
    if gp.exists():
        try:
            from pygltflib import GLTF2
            m = GLTF2().load_binary(str(gp))
            extras = (m.scenes[0].extras or {}) if m.scenes and m.scenes[0].extras else {}
            g = extras.get('k3d_garden') or {}
            if 'knowledge_sectors' in g:
                sectors = {k: tuple(v) for k, v in (g.get('knowledge_sectors') or {}).items()}
            trees = g.get('trees') or []
        except Exception:
            pass
    make_svg(sectors, Path(args.out), radius=200.0, trees=trees)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

