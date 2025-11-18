import time
import numpy as np

from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.cranium.sovereign.trm_ternary_launcher import TRMTernaryLauncher


def benchmark():
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32) * 0.01
    W2 = rng.standard_normal((512, 1024), dtype=np.float32) * 0.01
    W3 = rng.standard_normal((1024, 512), dtype=np.float32) * 0.01
    W4 = rng.standard_normal((512, 1024), dtype=np.float32) * 0.01
    q = rng.standard_normal(512, dtype=np.float32)
    y = np.zeros(512, dtype=np.float32)
    z = np.zeros(512, dtype=np.float32)

    base = TRMLauncher()
    ternary = TRMTernaryLauncher()

    def run(fn):
        times = []
        for _ in range(10):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1e6)
        return np.median(times)

    t_base = run(lambda: base.refine(q, y, z, W1, W2, W3, W4, n_steps=6))
    t_ternary = run(lambda: ternary.refine(q, y, z, W1, W2, W3, W4, n_steps=6))

    print(f"Baseline median us: {t_base:.2f}")
    print(f"Ternary median us: {t_ternary:.2f}")
    print(f"Speedup: {t_base / t_ternary:.2f}x")


if __name__ == "__main__":
    benchmark()
