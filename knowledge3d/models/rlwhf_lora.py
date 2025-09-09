from __future__ import annotations

"""
RLWHF LoRA Trainer for TinyLlama (or other HF causal LMs).

Fine-tunes a base model with LoRA or QLoRA using the RLWHF dataset rows
{query, answer, contexts[], reward}. Saves only the PEFT adapter plus a
base_model_id.txt to load at inference time.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.models.rlwhf_lora \
    --dataset docs/reports/training/rlwhf_dataset.jsonl \
    --out ../Knowledge3D.local/models/rlwhf_lora_tinyllama \
    --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 1 --batch 2 --max_len 384 --lr 2e-4 --qlora
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Missing dependency 'peft': {e}")


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
        ctx_lines: List[str] = []
        for t in ctxs[:max_ctx]:
            t = str(t or "").strip()
            if not t:
                continue
            if len(t) > 300:
                t = t[:297] + "..."
            ctx_lines.append(f"- {t}")
        sys = (
            "You are K3D's integrated LLM skill. You must answer using ONLY the provided contexts from the House memory. "
            "Do not invent facts. If information is missing, explicitly say 'I don't know' and suggest where to look next (label/room)."
        )
        prompt = sys + "\n\n" + ("Context:\n" + "\n".join(ctx_lines) + "\n\n" if ctx_lines else "") + f"Question: {q}\n\nAnswer:"
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
        full = s.prompt + " " + s.answer
        enc = self.tok(full, truncation=True, max_length=self.max_len, return_tensors="pt")
        ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        enc_p = self.tok(s.prompt, truncation=True, max_length=self.max_len, return_tensors="pt")
        plen = int(enc_p["input_ids"].shape[1])
        labels = ids.clone()
        labels[: min(plen, labels.shape[0])] = -100
        return {"input_ids": ids, "attention_mask": attn, "labels": labels, "weight": torch.tensor(s.weight, dtype=torch.float32)}


def collate_pad(batch: List[dict], pad_id: int) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    def pad(t: torch.Tensor, val: int) -> torch.Tensor:
        if t.shape[0] == max_len:
            return t
        pad = torch.full((max_len - t.shape[0],), val, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)
    input_ids = torch.stack([pad(b["input_ids"], pad_id) for b in batch])
    attention_mask = torch.stack([pad(b["attention_mask"], 0) for b in batch])
    labels = torch.stack([pad(b["labels"], -100) for b in batch])
    weights = torch.stack([b["weight"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": weights}


def train_lora(dataset: Path, out: Path, base: str, epochs: int = 1, batch: int = 2, max_len: int = 384, lr: float = 2e-4, qlora: bool = True) -> dict:
    from transformers import BitsAndBytesConfig  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb_cfg = None
    if qlora:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb_cfg, torch_dtype=(torch.float16 if torch.cuda.is_available() else None), device_map="auto" if qlora else None)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.train()
    samples = build_samples(dataset)
    if not samples:
        raise SystemExit(f"No samples in {dataset}")
    ds = RLWHFDataset(samples, tok, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=lambda b: collate_pad(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    total_steps = epochs * max(1, len(dl))
    step = 0
    for ep in range(epochs):
        for batch_data in dl:
            step += 1
            opt.zero_grad(set_to_none=True)
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            weights = batch_data["weights"].to(device)
            outp = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outp.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_tok = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_tok = loss_tok.view(shift_labels.size(0), shift_labels.size(1))
            mask = (shift_labels != -100).float()
            loss_sample = (loss_tok * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            loss = (loss_sample * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    out.mkdir(parents=True, exist_ok=True)
    # Save adapter and base model id
    model.save_pretrained(str(out))
    (out / "base_model_id.txt").write_text(base, encoding="utf-8")
    info = {"samples": len(samples), "epochs": epochs, "batch": batch, "base": base, "qlora": bool(qlora)}
    (out / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train RLWHF LoRA adapter")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--qlora", action="store_true")
    args = ap.parse_args()
    info = train_lora(Path(args.dataset), Path(args.out), base=str(args.base), epochs=int(args.epochs), batch=int(args.batch), max_len=int(args.max_len), lr=float(args.lr), qlora=bool(args.qlora))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

