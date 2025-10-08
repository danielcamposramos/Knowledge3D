# DEPRECATED: legacy pre-PTX script; kept for reference. Outputs belong in Knowledge3D.local/old_attempts.
"""
GPU smoke test: verifies PyTorch CUDA availability and runs a tiny HF forward/backward loop.

Outputs
- Prints torch + CUDA info
- Allocates a small tensor on GPU and measures matmul time
- Runs a miniature transformer sequence classification model for a few steps

Usage
  python3 -m knowledge3d.tools.gpu_smoke
"""
from __future__ import annotations

import time


def main() -> None:  # pragma: no cover
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return
    print("device count:", torch.cuda.device_count())
    print("device name:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"mem before: free={free/1e6:.0f}MB total={total/1e6:.0f}MB")
    # Matmul smoke
    a = torch.randn(2048, 2048, device='cuda')
    b = torch.randn(2048, 2048, device='cuda')
    t0 = time.perf_counter(); c = a @ b; torch.cuda.synchronize(); t1 = time.perf_counter()
    print(f"matmul 2048x2048: {(t1-t0)*1000:.1f} ms, result norm={c.norm().item():.2f}")
    del a, b, c
    # Tiny HF loop
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        print("transformers not installed:", e)
        return
    tok = AutoTokenizer.from_pretrained('distilroberta-base')
    mdl = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=3).to('cuda')
    texts = ["goto library", "move left 2", "teleport to [1,2,3]"] * 8
    batch = tok(texts, return_tensors='pt', padding=True, truncation=True).to('cuda')
    labels = torch.randint(0, 3, (len(texts),), device='cuda')
    opt = torch.optim.AdamW(mdl.parameters(), lr=3e-4)
    t0 = time.perf_counter()
    mdl.train()
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        out = mdl(**batch, labels=labels)
        out.loss.backward()
        opt.step()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    print(f"mini-train 10 steps: {(t1-t0):.2f} s, final loss={out.loss.item():.3f}")
    free2, total2 = torch.cuda.mem_get_info()
    print(f"mem after: free={free2/1e6:.0f}MB total={total2/1e6:.0f}MB")


if __name__ == '__main__':
    main()

