import argparse
from typing import Dict, Any

import numpy as np

from knowledge3d.cranium.embedding_generator import DynamicEmbeddingGenerator
from knowledge3d.tools.phase2.export_tree import exportTreeToGLB


def _generate_tree_py(emb: np.ndarray) -> Dict[str, Any]:
    """Lightweight Python version of FractalTreeGenerator to avoid C++ binding.
    Produces a small 3-level tree with 3 children per node, mutating embeddings.
    """
    def mutate(e: np.ndarray, depth: int, seed: int) -> np.ndarray:
        rs = np.random.RandomState(depth * 1337 + seed)
        return (e + rs.normal(0, 0.05 * max(1, depth), size=e.shape)).astype(np.float32)

    def node(pos, e):
        return {"position": [float(pos[0]), float(pos[1]), float(pos[2])], "embedding": e.tolist(), "children": [], "honesty": float(e[72]) if e.shape[0] > 72 else 1.0}

    angle = float(emb[0]) * np.pi if emb.shape[0] > 0 else 0.5
    length = float(0.5 + emb[1]) if emb.shape[0] > 1 else 1.0
    depth = int(np.clip(emb[2] * 4.0 + 1.0, 1.0, 5.0)) if emb.shape[0] > 2 else 3

    root = node((0.0, 0.0, 0.0), emb)
    frontier = [(root, (0.0, 0.0, 0.0), emb, 1)]
    while frontier:
        parent, ppos, pemb, lvl = frontier.pop(0)
        if lvl >= depth:
            continue
        for i, a in enumerate([angle, 0.0, -angle]):
            dx, dy, dz = np.sin(a) * length, np.cos(a) * length, 0.4
            cpos = (ppos[0] + dx, ppos[1] + dy, ppos[2] + dz)
            cemb = mutate(pemb, lvl, i)
            child = node(cpos, cemb)
            parent["children"].append(child)
            frontier.append((child, cpos, cemb, lvl + 1))
    return root


class GrowTreeCommand:
    def __init__(self, garden_path: str = "viewer/public/knowledge_garden.glb"):
        self.garden_path = garden_path
        self.embedding_generator = DynamicEmbeddingGenerator()

    def execute(self, domain: str, context: dict | None = None) -> str:
        emb = self.embedding_generator.generate(domain, 'text')
        if emb.shape[0] > 72:
            emb = emb.copy(); emb[72] = 1.0
        tree = _generate_tree_py(emb)
        ok = exportTreeToGLB(tree, self.garden_path)
        return f"🌳 Tree '{domain}' grown in Knowledge Garden." if ok else "Failed to grow tree."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", default="viewer/public/knowledge_garden.glb")
    args = ap.parse_args()
    cmd = GrowTreeCommand(args.out)
    print(cmd.execute(args.domain))


if __name__ == "__main__":
    main()

