from knowledge3d.cranium.specialists.batch_optimizer import BatchOptimizer


def test_batch_optimizer_suggests_increase():
    opt = BatchOptimizer(target_utilization=0.75, max_vram_mb=180.0)
    new_bs = opt.suggest_batch_size(current_batch_size=32, gpu_utilization=0.07, vram_used_mb=108.0)
    assert new_bs >= 32
    assert new_bs <= 256
    assert new_bs % 8 == 0
