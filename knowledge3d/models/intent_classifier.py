from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


LABELS = [
    # navigation
    "teleport", "move", "goto", "follow", "orbit",
    # exploration
    "show", "find_related", "expand", "hide",
    # interaction
    "touch", "talk", "give",
]


def synthesize_examples(n_per_label: int = 100) -> tuple[list[str], list[str]]:
    import random
    X: list[str] = []
    y: list[str] = []
    tele_t = [
        "teleport to [%.1f,%.1f,%.1f]",
        "teleport to [%d,%d,%d]",
        "teleport to [%d, %d, %d]",
    ]
    move_t = ["move %s %d", "move %s %.1f", "go %s %d units"]
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
        X.append(random.choice(move_t) % (random.choice(dirs), random.randint(1,5)))
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
        # find
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


def _iter_logs(log_dir: Path) -> Iterable[dict]:
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


def build_dataset(log_dir: Path) -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    # Use chat_response entries to create supervised examples
    for rec in _iter_logs(log_dir):
        if rec.get("type") == "chat_response":
            text = rec.get("text") or ""
            resp = rec.get("response") or {}
            act_type = resp.get("type")
            action = resp.get("action")
            if act_type in {"navigation", "exploration", "interaction"} and action in LABELS:
                X.append(str(text))
                y.append(str(action))
    return X, y


def train_model(log_dir: Path, out_path: Path, synth_per_label: int = 0) -> dict:
    X, y = build_dataset(log_dir)
    if synth_per_label and synth_per_label > 0:
        Xs, ys = synthesize_examples(synth_per_label)
        X += Xs
        y += ys
    if not X:
        raise RuntimeError(f"No training examples found in {log_dir}")
    # Use stratify only when every class has >=2 samples
    from collections import Counter
    cnt = Counter(y)
    use_strat = all(v >= 2 for v in cnt.values()) and (len(set(y)) > 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if use_strat else None
    )
    pipe = Pipeline([
        ("vect", CountVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=1)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    report = classification_report(y_te, y_pred, zero_division=0, output_dict=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    return {"samples": len(X), "classes": sorted(set(y)), "report": report, "model": str(out_path)}


def train_from_templates(templates_dir: Path, langs: list[str], out_path: Path, n_per_label: int = 200) -> dict:
    """Train a compact multilingual intent model using YAML templates.

    Reuses the synthetic templating from the HF pipeline to avoid heavy dependencies.
    """
    from .intent_hf import synthesize_examples_from_templates  # type: ignore
    files = [templates_dir / f"{ln}.yaml" for ln in langs if (templates_dir / f"{ln}.yaml").exists()]
    if not files:
        raise RuntimeError(f"No template files found for langs={langs} in {templates_dir}")
    X, y = synthesize_examples_from_templates(files, n_per_label)
    if not X:
        raise RuntimeError("No examples synthesized from templates")
    pipe = Pipeline([
        ("vect", CountVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=1)),
    ])
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    report = classification_report(y_te, y_pred, zero_division=0, output_dict=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    return {"samples": len(X), "classes": sorted(set(y)), "report": report, "model": str(out_path)}


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def predict_action(model: Pipeline, text: str) -> Tuple[str, float]:
    proba = model.predict_proba([text])[0]
    idx = int(proba.argmax())
    label = model.classes_[idx]
    conf = float(proba[idx])
    return label, conf


def main():  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Train/evaluate a simple intent-action classifier from live logs or templates")
    parser.add_argument("command", choices=["train", "train-templates", "predict"], help="Mode")
    parser.add_argument("--logs", default=str((Path(__file__).resolve().parents[2].parent / (Path(__file__).resolve().parents[2].name + ".local") / "logs")), help="Logs directory")
    parser.add_argument("--model", default=str((Path(__file__).resolve().parents[2].parent / (Path(__file__).resolve().parents[2].name + ".local") / "models" / "intent.pkl")), help="Model path")
    parser.add_argument("--synth-per-label", type=int, default=0, help="Synthesize N examples per label to augment logs")
    parser.add_argument("--text", help="Text to predict in predict mode")
    parser.add_argument("--templates-dir", default=str((Path(__file__).resolve().parents[2] / "data" / "intent_templates")))
    parser.add_argument("--langs", default="en,pt,es")
    args = parser.parse_args()
    logs = Path(args.logs)
    model_path = Path(args.model)
    if args.command == "train":
        res = train_model(logs, model_path, synth_per_label=args.synth_per_label)
        print(json.dumps(res, indent=2))
    elif args.command == "train-templates":
        res = train_from_templates(Path(args.templates_dir), [s.strip() for s in str(args.langs).split(',') if s.strip()], model_path, n_per_label=args.synth_per_label or 200)
        print(json.dumps(res, indent=2))
    else:
        if not args.text:
            parser.error("--text required for predict mode")
        model = load_model(model_path)
        label, conf = predict_action(model, args.text)
        print(json.dumps({"action": label, "confidence": conf}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
