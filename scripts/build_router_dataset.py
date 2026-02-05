#!/usr/bin/env python3
"""
Build a training dataset for the Theorem Router.

Class 0: General Solver (GSM8K)
Class 1: Calculus Specialist (Microbench)
"""

import argparse
import json
import random
from pathlib import Path

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return data

def main():
    parser = argparse.ArgumentParser(description="Build Router Dataset")
    parser.add_argument("--calc", required=True, help="Path to calculus microbench (Class 1)")
    parser.add_argument("--gsm8k", required=True, help="Path to GSM8K train.jsonl (Class 0)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--balance", action="store_true", help="Undersample GSM8K to match Calc size")
    args = parser.parse_args()

    # Load Class 1: Calculus
    calc_data = load_jsonl(args.calc)
    print(f"Loaded {len(calc_data)} calculus problems.")

    # Load Class 0: GSM8K
    gsm_data = load_jsonl(args.gsm8k)
    print(f"Loaded {len(gsm_data)} GSM8K problems.")

    # Prepare entries
    dataset = []

    for item in calc_data:
        text = item.get("problem", item.get("question", ""))
        if text:
            dataset.append({"text": text, "label": 1, "source": "calculus"})

    # Undersample GSM8K if requested
    if args.balance:
        gsm_sample = random.sample(gsm_data, min(len(gsm_data), len(calc_data)))
    else:
        gsm_sample = gsm_data

    for item in gsm_sample:
        text = item.get("question", item.get("problem", ""))
        if text:
            dataset.append({"text": text, "label": 0, "source": "gsm8k"})

    random.shuffle(dataset)

    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Wrote {len(dataset)} entries to {args.output}")

if __name__ == "__main__":
    main()