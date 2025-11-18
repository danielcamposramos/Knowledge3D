import numpy as np

from knowledge3d.cranium.sleep.knowledge_sleep import KnowledgeSleepCycle


class FakeDepth:
    def __init__(self, trits):
        self.trits = trits

    def compute(self, embeddings, query, attract_thresh=0.35, repel_thresh=-0.05):
        n = len(self.trits)
        n_words = (n + 15) // 16
        out = np.zeros(n_words, dtype=np.uint32)
        for i, t in enumerate(self.trits):
            bits = 2 if t == 1 else (1 if t == 0 else 0)
            word = i >> 4
            shift = (i & 0xF) << 1
            out[word] |= np.uint32(bits << shift)
        return out


class FakePruner:
    def __init__(self, keep_mask):
        self.keep_mask = keep_mask

    def decide(self, scores, keep_thresh=0.5, drop_thresh=0.05):
        return np.array([1 if k else -1 for k in self.keep_mask], dtype=np.int8)


def test_knowledge_sleep_depth_and_prune(tmp_path):
    # Three stars: keep first two, drop third by depth/prune
    stars = [{"metadata": {}} for _ in range(3)]
    embeddings = [np.array([1.0, 0.0], dtype=np.float32),
                  np.array([0.5, 0.0], dtype=np.float32),
                  np.array([-1.0, 0.0], dtype=np.float32)]
    depth = FakeDepth([1, 1, -1])  # drop last
    pruner = FakePruner([True, False])  # within cluster drop second
    ks = KnowledgeSleepCycle(tmp_path / "galaxy.pkl", tmp_path / "house.glb", rpn_engine=None, depth_computer=depth, pruner=pruner)  # type: ignore
    ks.galaxy_stars = stars
    ks.star_embeddings = embeddings
    clusters = ks.cluster_stars_rpn(n_clusters=2)
    assert all(2 not in c for c in clusters)  # depth removed last star
    # Materialize first cluster using two items; pruner removes second
    house_obj = ks.materialize_cluster(clusters[0], cluster_id=0)
    assert house_obj["cluster_size"] == 1
