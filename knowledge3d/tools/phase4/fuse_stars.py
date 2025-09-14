import argparse
from typing import List

from .drag_drop import DragDropHandler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy', default='viewer/public/galaxy_registry.json')
    ap.add_argument('--workshop', default='viewer/public/workshop/workshop_registry.json')
    ap.add_argument('star_ids', nargs='+', help='Workshop star IDs to fuse')
    args = ap.parse_args()
    h = DragDropHandler(args.galaxy, args.workshop)
    fused = h.fuse_stars(args.star_ids)
    if fused is None:
        print('fuse failed')
    else:
        print(f"fused id={fused['id']} shape={fused.get('shape_type')} dims={len(fused.get('embedding',[]))}")


if __name__ == '__main__':
    main()

