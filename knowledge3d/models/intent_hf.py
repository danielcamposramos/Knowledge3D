from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    # Block unsafe native runs on Debian unless explicitly allowed
    from ..utils.env_guard import enforce_containment  # type: ignore
    enforce_containment("HF intent training")
except Exception:
    pass

# Labels align with EnhancedChatProcessor actions
INTENT_LABELS: List[str] = [
    "teleport",
    "move",
    "goto",
    "follow",
    "orbit",
    "show",
    "find_related",
    "expand",
    "hide",
    "touch",
    "talk",
    "give",
]


def _iter_logs(log_dir: Path):
    for p in sorted(log_dir.glob("session-*.jsonl")):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except OSError:
            continue


def build_dataset_from_logs(log_dir: Path) -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    seen = set()
    for rec in _iter_logs(log_dir):
        if rec.get("type") == "chat_response":
            text = str(rec.get("text") or "")
            resp = rec.get("response") or {}
            act_type = resp.get("type")
            action = resp.get("action")
            if act_type in {"navigation", "exploration", "interaction"} and action in INTENT_LABELS:
                key = (text, action)
                if key in seen:
                    continue
                seen.add(key)
                X.append(text)
                y.append(action)
    return X, y


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("pyyaml not installed. pip install pyyaml") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def synthesize_examples_from_templates(files: List[Path], n_per_label: int = 100) -> Tuple[List[str], List[str]]:
    import random
    X: List[str] = []
    y: List[str] = []
    packs = [_load_yaml(f) for f in files]
    for pack in packs:
        intents = pack.get("intents", {})
        vocab = pack.get("vocab", {})
        dirs = vocab.get("directions", ["left", "right"])  # fallbacks
        places = vocab.get("places", ["market"]) 
        topics = vocab.get("topics", ["graphs"]) 
        objects = vocab.get("objects", ["door"]) 
        persons = vocab.get("persons", ["guide"]) 
        items = vocab.get("items", ["key"]) 
        for _ in range(n_per_label):
            for intent, patterns in intents.items():
                pat = random.choice(patterns)
                s = pat
                # fill variables
                if "{x}" in s:
                    s = s.replace("{x}", str(random.randint(-5, 5)))
                if "{y}" in s:
                    s = s.replace("{y}", str(random.randint(-5, 5)))
                if "{z}" in s:
                    s = s.replace("{z}", str(random.randint(-5, 5)))
                if "{direction}" in s:
                    s = s.replace("{direction}", random.choice(dirs))
                if "{distance}" in s:
                    s = s.replace("{distance}", str(random.randint(1, 5)))
                if "{place}" in s:
                    s = s.replace("{place}", random.choice(places))
                if "{topic}" in s:
                    s = s.replace("{topic}", random.choice(topics))
                if "{object}" in s:
                    s = s.replace("{object}", random.choice(objects))
                if "{person}" in s:
                    s = s.replace("{person}", random.choice(persons))
                if "{item}" in s:
                    s = s.replace("{item}", random.choice(items))
                X.append(s)
                y.append(intent)
    return X, y


def synthesize_examples(n_per_label: int = 100) -> Tuple[List[str], List[str]]:
    import random
    X: List[str] = []
    y: List[str] = []
    # Simple templated phrases per label
    tele_t = [
        "teleport to [%.1f,%.1f,%.1f]",
        "teleport to [%d,%d,%d]",
        "teleport to [%d, %d, %d]",
    ]
    move_t = [
        "move %s %d",
        "move %s %.1f",
        "go %s %d units",
    ]
    goto_t = ["goto %s", "go to %s", "navigate to %s"]
    follow_t = ["follow %s", "follow agent %s", "follow user %s"]
    orbit_t = ["orbit around %s", "circle around %s"]
    show_t = ["show me %s", "display %s", "reveal %s"]
    find_t = ["find related to %s", "related to %s", "show related %s"]
    expand_t = ["expand %s", "expand area %s", "expand cluster %s"]
    hide_t = ["hide %s", "conceal %s", "hide the %s"]
    touch_t = ["touch %s", "tap %s", "poke %s"]
    talk_t = ["talk to %s", "speak to %s", "ask %s"]
    give_t = ["give %s to %s", "hand %s to %s", "offer %s to %s"]
    dirs = ["left", "right", "up", "down", "forward", "back"]
    nouns = [
        "library", "market", "gatekeeper", "door", "portal", "graph", "cluster",
        "quantum", "embeddings", "sorting algorithms", "legend", "tooltip",
        "blue cube", "red sphere", "avatar alice", "bob", "guide",
    ]

    for _ in range(n_per_label):
        # teleport
        r = random.random()
        if r < 0.33:
            X.append(tele_t[0] % (random.uniform(-5,5), random.uniform(-5,5), random.uniform(-5,5)))
        elif r < 0.66:
            X.append(tele_t[1] % (random.randint(-5,5), random.randint(-5,5), random.randint(-5,5)))
        else:
            X.append(tele_t[2] % (random.randint(-5,5), random.randint(-5,5), random.randint(-5,5)))
        y.append("teleport")
        # move
        t = random.choice(move_t)
        X.append(t % (random.choice(dirs), random.randint(1,5)))
        y.append("move")
        # goto
        X.append(random.choice(goto_t) % random.choice(nouns))
        y.append("goto")
        # follow
        X.append(random.choice(follow_t) % random.choice(["agent one","guide","alice"]))
        y.append("follow")
        # orbit
        X.append(random.choice(orbit_t) % random.choice(["Saturn","hub","core"]))
        y.append("orbit")
        # show
        X.append(random.choice(show_t) % random.choice(["graphs","math","physics"]))
        y.append("show")
        # find_related
        X.append(random.choice(find_t) % random.choice(["clustering","neural networks","embeddings"]))
        y.append("find_related")
        # expand
        X.append(random.choice(expand_t) % random.choice(["cluster physics","area quantum","region math"]))
        y.append("expand")
        # hide
        X.append(random.choice(hide_t) % random.choice(["legend","tooltip","panel"]))
        y.append("hide")
        # touch
        X.append(random.choice(touch_t) % random.choice(["blue cube","red sphere","door"]))
        y.append("touch")
        # talk
        X.append(random.choice(talk_t) % random.choice(["avatar alice","guide","agent"]))
        y.append("talk")
        # give
        X.append(random.choice(give_t) % (random.choice(["key","token","coin"]), random.choice(["bob","clerk","gatekeeper"])))
        y.append("give")
    return X, y


def train_from_logs_and_synth(
    log_dir: Path,
    out_dir: Path,
    base_model: str = "xlm-roberta-base",
    epochs: int = 3,
    synth_per_label: int = 100,
    lr: float = 5e-5,
    batch_size: int = 16,
    templates_dir: Optional[Path] = None,
    langs: Optional[List[str]] = None,
) -> Dict[str, object]:
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    Xl, yl = build_dataset_from_logs(log_dir)
    if templates_dir and langs:
        files = [templates_dir / f"{ln}.yaml" for ln in langs if (templates_dir / f"{ln}.yaml").exists()]
        Xs, ys = synthesize_examples_from_templates(files, synth_per_label)
    else:
        Xs, ys = synthesize_examples(synth_per_label)
    X = Xl + Xs
    y = yl + ys
    if not X:
        raise RuntimeError("No training data (logs + synth) produced")
    # map labels to ids
    label2id = {lbl: i for i, lbl in enumerate(INTENT_LABELS)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    y_ids = [label2id.get(lbl, 0) for lbl in y]
    tok = AutoTokenizer.from_pretrained(base_model)
    ds = Dataset.from_dict({"text": X, "label": y_ids})

    def tokenize(examples):
        return tok(examples["text"], truncation=True, padding=False)

    ds = ds.shuffle(seed=42)
    n = len(ds)
    n_te = max(100, int(0.1 * n))
    ds_te = ds.select(range(n_te))
    ds_tr = ds.select(range(n_te, n))
    ds_tr = ds_tr.map(tokenize, batched=True)
    ds_te = ds_te.map(tokenize, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    ds_tr = ds_tr.remove_columns([c for c in ds_tr.column_names if c not in cols])
    ds_te = ds_te.remove_columns([c for c in ds_te.column_names if c not in cols])

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(INTENT_LABELS), id2label=id2label, label2id=label2id
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "runs"),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True if False else False,
    )
    import evaluate
    acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc.compute(predictions=preds, references=labels)

    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_te, tokenizer=tok, compute_metrics=compute_metrics)
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "labels.json").write_text(json.dumps({"labels": INTENT_LABELS}), encoding="utf-8")
    return {"samples": len(X), "log_samples": len(Xl), "synth_samples": len(Xs), "model_dir": str(out_dir)}


@dataclass
class HFModel:
    tok: object
    mdl: object
    labels: List[str]

    def predict_proba(self, text: str) -> Tuple[str, float]:
        import torch
        with torch.no_grad():
            inputs = self.tok(text, return_tensors="pt")
            logits = self.mdl(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(probs.argmax())
            return self.labels[idx], float(probs[idx])


def load_model(model_dir: Path) -> HFModel:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    labels = INTENT_LABELS
    lab_path = model_dir / "labels.json"
    if lab_path.exists():
        try:
            labels = json.loads(lab_path.read_text(encoding="utf-8")).get("labels", labels)
        except Exception:
            pass
    return HFModel(tok=tok, mdl=mdl, labels=labels)


def predict_action(model: HFModel, text: str) -> Tuple[str, float]:
    return model.predict_proba(text)


def main():  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Train/predict HF intent model with logs + synthetic data")
    p.add_argument("command", choices=["train", "predict"]) 
    p.add_argument("--logs", default=str((Path(__file__).resolve().parents[2].parent / (Path(__file__).resolve().parents[2].name + ".local") / "logs")))
    p.add_argument("--out", default=str((Path(__file__).resolve().parents[2].parent / (Path(__file__).resolve().parents[2].name + ".local") / "models" / "intent_hf")))
    p.add_argument("--pretrained", default="xlm-roberta-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--synth-per-label", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--templates-dir", default=str((Path(__file__).resolve().parents[2] / "data" / "intent_templates")))
    p.add_argument("--langs", default="en,pt,es")
    p.add_argument("--text", help="text to predict")
    args = p.parse_args()
    if args.command == "train":
        # Configure FP16 if requested
        global TrainingArguments
        if args.fp16:
            # Rebind TrainingArguments to include fp16=True via a thin wrapper
            from transformers import TrainingArguments as _TA  # type: ignore
            class _TA2(_TA):
                def __init__(self, *a, **kw):
                    kw.setdefault("fp16", True)
                    super().__init__(*a, **kw)
            TrainingArguments = _TA2  # type: ignore
        res = train_from_logs_and_synth(
            Path(args.logs),
            Path(args.out),
            base_model=args.pretrained,
            epochs=args.epochs,
            synth_per_label=args.synth_per_label,
            batch_size=args.batch_size,
            templates_dir=Path(args.templates_dir),
            langs=[s.strip() for s in str(args.langs).split(',') if s.strip()],
        )
        print(json.dumps(res, indent=2))
    else:
        if not args.text:
            p.error("--text required for predict")
        model = load_model(Path(args.out))
        lab, conf = predict_action(model, args.text)
        print(json.dumps({"action": lab, "confidence": conf}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
