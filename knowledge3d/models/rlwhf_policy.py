from __future__ import annotations

"""
RLWHF Policy Trainer — reward‑weighted SFT for answer generation.

Trains a small causal LM (default: distilgpt2) on RLWHF JSONL rows
with fields: {query, answer, contexts[], reward}.

Loss: token‑level cross‑entropy on the Answer segment only, weighted per‑sample
by a monotonic mapping of the scalar reward into [0.1, 1.0]. This is a simple
and robust RWSF baseline that runs on a single consumer GPU.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.models.rlwhf_policy \
    --dataset docs/reports/training/rlwhf_dataset.jsonl \
    --out ../Knowledge3D.local/models/rlwhf_policy \
    --model distilgpt2 --epochs 1 --batch 4 --max_len 384
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def reward_to_weight(r: float) -> float:
    # Map reward roughly in [-0.25, 1.5] to [0.1, 1.0]
    lo, hi = -0.25, 1.5
    r = max(lo, min(hi, float(r)))
    x = (r - lo) / (hi - lo)
    return float(0.1 + 0.9 * x)


@dataclass
class Sample:
    prompt: str
    answer: str
    weight: float


def build_samples(dataset_path: Path, max_ctx: int = 4) -> List[Sample]:
    out: List[Sample] = []
    for rec in iter_jsonl(dataset_path):
        q = str(rec.get("query") or "").strip()
        a = str(rec.get("answer") or "").strip()
        ctxs = rec.get("contexts") or []
        rw = float(rec.get("reward") or 0.0)
        if not q or not a:
            continue
        # Compose a grounded prompt consistent with our LLM skill
        ctx_lines: List[str] = []
        for t in ctxs[:max_ctx]:
            t = str(t or "").strip()
            if not t:
                continue
            if len(t) > 300:
                t = t[:297] + "..."
            ctx_lines.append(f"- {t}")
        sys = (
            "You are K3D's integrated LLM skill. You must answer using ONLY the provided "
            "contexts from the House memory. Do not invent facts. If information is missing, "
            "explicitly say 'I don't know' and suggest where to look next (label/room). "
            "When appropriate, cite labels you used in parentheses, e.g., (sources: A, B)."
        )
        prompt = (
            sys + "\n\n" +
            ("Context (ground truth; use only this):\n" + "\n".join(ctx_lines) + "\n\n" if ctx_lines else "") +
            f"Question: {q}\n\nAnswer:"
        )
        out.append(Sample(prompt=prompt, answer=a, weight=reward_to_weight(rw)))
    return out


class RLWHFDataset(Dataset):
    def __init__(self, samples: List[Sample], tok: AutoTokenizer, max_len: int = 384) -> None:
        self.samples = samples
        self.tok = tok
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # Tokenize prompt+answer; labels mask prompt tokens with -100
        full = s.prompt + " " + s.answer
        enc = self.tok(full, truncation=True, max_length=self.max_len, return_tensors="pt")
        ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        # Find the index where answer starts; simple approach: tokenize prompt alone
        enc_p = self.tok(s.prompt, truncation=True, max_length=self.max_len, return_tensors="pt")
        plen = int(enc_p["input_ids"].shape[1])
        labels = ids.clone()
        labels[: min(plen, labels.shape[0])] = -100
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "weight": torch.tensor(s.weight, dtype=torch.float32),
        }


def collate_pad(batch: List[dict], pad_id: int) -> dict:
    # Pad to the max length in the batch
    max_len = max(item["input_ids"].shape[0] for item in batch)
    def pad_tensor(t: torch.Tensor, value: int) -> torch.Tensor:
        if t.shape[0] == max_len:
            return t
        pad = torch.full((max_len - t.shape[0],), value, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)
    input_ids = torch.stack([pad_tensor(b["input_ids"], pad_id) for b in batch])
    attention_mask = torch.stack([pad_tensor(b["attention_mask"], 0) for b in batch])
    labels = torch.stack([pad_tensor(b["labels"], -100) for b in batch])
    weights = torch.stack([b["weight"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": weights}


def train(dataset_path: Path, out_dir: Path, model_id: str = "distilgpt2", epochs: int = 1, batch_size: int = 4, max_len: int = 384, lr: float = 5e-5) -> dict:
    # Enforce GPU when K3D_STRICT_GPU is set
    import os as _os
    strict = _os.getenv("K3D_STRICT_GPU", "0").strip() not in {"", "0", "false", "False"}
    if strict and not torch.cuda.is_available():
        raise SystemExit("GPU required (K3D_STRICT_GPU=1) but CUDA is not available for RLWHF policy training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        # Set pad token to eos for GPT2 family
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    samples = build_samples(dataset_path)
    if not samples:
        raise SystemExit(f"No samples in {dataset_path}")
    ds = RLWHFDataset(samples, tok, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_pad(b, tok.pad_token_id))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * max(1, len(dl))
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=max(1, total_steps//10), num_training_steps=total_steps)
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    model.train()
    step = 0
    for ep in range(epochs):
        for batch in dl:
            step += 1
            optim.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [B, T, V]
            # Shift for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Compute per-token loss
            loss_tok = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_tok = loss_tok.view(shift_labels.size(0), shift_labels.size(1))
            mask = (shift_labels != -100).float()
            # Per-sample mean over valid tokens
            loss_sample = (loss_tok * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            # Apply reward weights
            loss = (loss_sample * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); scheduler.step()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    info = {"samples": len(samples), "epochs": epochs, "batch": batch_size, "model": model_id}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train RLWHF policy via reward-weighted SFT")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()
    info = train(Path(args.dataset), Path(args.out), model_id=str(args.model), epochs=int(args.epochs), batch_size=int(args.batch), max_len=int(args.max_len), lr=float(args.lr))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
