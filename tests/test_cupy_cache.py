#!/usr/bin/env python3
"""Test if CuPy cache is working properly."""

def main() -> int:
    import os
    from pathlib import Path

    # Point to the pre-compiled cache
    cache_dir = Path(__file__).parent / ".cupy_cache"
    os.environ["CUPY_CACHE_DIR"] = str(cache_dir)

    print(f"Testing CuPy with cache: {cache_dir}")
    print(f"Cache exists: {cache_dir.exists()}")

    try:
        import cupy as cp

        print("✓ CuPy imported successfully")
        print(f"  CUDA available: {cp.cuda.is_available()}")
        print(f"  Device: {cp.cuda.Device().id}")
    except Exception as e:
        print(f"✗ Failed to import CuPy: {e}")
        return 1

    # Test the exact operations that SemanticNavigator needs
    tests: list[bool] = []

    print("\n1. Testing cp.linalg.norm (critical for LED-A*)...")
    try:
        # This is what led_pathfinder.py:142 does
        src_pos = cp.random.rand(100, 3).astype(cp.float32)
        dst_pos = cp.random.rand(100, 3).astype(cp.float32)
        distances = cp.linalg.norm(src_pos - dst_pos, axis=1)
        print(f"   ✓ linalg.norm succeeded: {distances.shape}")
        tests.append(True)
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests.append(False)

    print("\n2. Testing cp.concatenate (critical for LED-A*)...")
    try:
        # This is what led_pathfinder.py:158-160 does
        a = cp.array([[1, 2], [3, 4]], dtype=cp.uint32)
        b = cp.array([[5, 6]], dtype=cp.uint32)
        c = cp.concatenate([a, b])
        print(f"   ✓ concatenate succeeded: {c.shape}")
        tests.append(True)
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests.append(False)

    print("\n3. Testing cp.argsort (critical for Morton octree)...")
    try:
        # This is what morton_octree.py:132 does
        morton_codes = cp.random.randint(0, 2**30, size=1000, dtype=cp.uint32)
        sorted_indices = cp.argsort(morton_codes)
        print(f"   ✓ argsort succeeded: {sorted_indices.shape}")
        tests.append(True)
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests.append(False)

    print("\n4. Testing cp.asarray and indexing...")
    try:
        edges_cpu = [[0, 1], [1, 2], [2, 3]]
        edges_gpu = cp.asarray(edges_cpu, dtype=cp.uint32)
        src_idx = edges_gpu[:, 0]
        dst_idx = edges_gpu[:, 1]
        print(
            "   ✓ asarray and indexing succeeded:"
            f" src={src_idx.shape}, dst={dst_idx.shape}"
        )
        tests.append(True)
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests.append(False)

    print("\n5. Testing boolean masking...")
    try:
        similarities = cp.random.rand(100).astype(cp.float32)
        threshold = 0.7
        mask = similarities > threshold
        filtered = similarities[mask]
        print(f"   ✓ Boolean masking succeeded: {filtered.shape}")
        tests.append(True)
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests.append(False)

    print("\n" + "=" * 60)
    if all(tests):
        print("✅ All CuPy operations succeeded!")
        print("The semantic navigator should work.")
        return 0

    print(f"❌ {tests.count(False)}/{len(tests)} tests failed")
    print("The semantic navigator will fail on this system.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
