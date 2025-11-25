"""Test spatial primitive detection across embedder modes."""

import numpy as np

from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor


def test_primitive_detection():
    """Test primitive detection with different embedders."""

    # Test grids
    grid_original = [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0],
    ]

    # Create transformations
    grid_rotate_90 = np.rot90(np.array(grid_original), k=-1).tolist()
    grid_rotate_180 = np.rot90(np.array(grid_original), k=2).tolist()
    grid_flip_h = np.fliplr(np.array(grid_original)).tolist()
    grid_flip_v = np.flipud(np.array(grid_original)).tolist()

    transformations = {
        "ROTATE_90": grid_rotate_90,
        "ROTATE_180": grid_rotate_180,
        "FLIP_H": grid_flip_h,
        "FLIP_V": grid_flip_v,
    }

    # Test with all embedder modes
    modes = ["procedural", "video", "audio", "multimodal"]

    print("=" * 60)
    print("Spatial Primitive Detection Test")
    print("=" * 60)

    results = {}

    for mode in modes:
        print(f"\n{mode.upper()} mode:")
        processor = ARCGridProcessor(matryoshka_dim=512, embedder_type=mode)

        mode_results = {}

        for transform_name, grid_transformed in transformations.items():
            detected = processor.detect_spatial_primitive(grid_original, grid_transformed)

            correct = detected["primitive"] == transform_name
            mode_results[transform_name] = {
                "detected": detected["primitive"],
                "correct": correct,
                "confidence": detected["confidence"],
            }

            status = "✅" if correct else "❌"
            print(
                f"  {status} {transform_name:15s}: "
                f"detected as {detected['primitive']:15s} "
                f"(conf: {detected['confidence']:.2f})"
            )

        results[mode] = mode_results

    # Summary
    print("\n" + "=" * 60)
    print("Detection Accuracy Summary")
    print("=" * 60 + "\n")

    for mode in modes:
        n_correct = sum(1 for r in results[mode].values() if r["correct"])
        n_total = len(results[mode])
        accuracy = n_correct / n_total * 100

        print(f"  {mode:15s}: {n_correct}/{n_total} correct ({accuracy:.1f}%)")

    print("\n✅ Primitive detection test complete!")

    return results


if __name__ == "__main__":
    test_primitive_detection()
